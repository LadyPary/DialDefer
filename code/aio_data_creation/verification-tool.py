#!/usr/bin/env python3
"""
Transcript Verifier (Flask, single-file)

Adds:
- Swap entire conversation speakers A<->B (Make Speaker B = OP)
- Loading indicator overlay for slow requests
- Add message above/below any message

Hotkeys:
- Ctrl+Z (Win/Linux) OR Command+Z (macOS) transcript-level undo (stack depth UNDO_DEPTH),
  only when NOT focused in a textarea (textarea undo remains native).
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, Response
import praw
import requests


# =========================
# CONFIG (edit these)
# =========================
WORKING_CSV = "clean_sweet_spot.csv"     # Your editable copy
VERIFIED_CSV = "VerifiedIDS.csv"                # One-column file: SubmissionID
IMAGE_CACHE_DIR = "image_cache"              # Will be created; per-submission subfolders
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
DEBUG = True

# Column names (must match your CSV)
COL_SUBMISSION_ID = "SubmissionID"
COL_INDEX = "Index"
COL_TRANSCRIPTION = "Transcription"

# Transcription shape
ALLOWED_SPEAKERS = {"Speaker A", "Speaker B"}

# PRAW / download behavior
ENABLE_PRAW_FETCH = True
DOWNLOAD_TIMEOUT_SECONDS = 20
DOWNLOAD_DELAY_SECONDS = 0.3
MAX_IMAGES_PER_SUBMISSION = 25

# UI defaults
DEFAULT_UNVERIFIED_ONLY = False
UNDO_DEPTH = 3


# =========================
# Flask app
# =========================
app = Flask(__name__)

_rows: List[Dict[str, str]] = []
_fieldnames: List[str] = []
_sid_to_rowidx: Dict[str, int] = {}
_index_to_sid: Dict[str, str] = {}
_verified: set[str] = set()

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
_NUM_IN_NAME = re.compile(r"(\d+)")


# =========================
# PRAW setup
# =========================
def make_reddit_client() -> praw.Reddit:
    """
    Uses existing .env keys exactly:
      client_id
      client_secret
      user_agent
      reddit-username
      password
    """
    load_dotenv()

    client_id = os.getenv("client_id", "").strip()
    client_secret = os.getenv("client_secret", "").strip()
    user_agent = os.getenv("user_agent", "").strip()
    username = os.getenv("reddit-username", "").strip()
    password = os.getenv("password", "").strip()

    missing = [k for k, v in [
        ("client_id", client_id),
        ("client_secret", client_secret),
        ("user_agent", user_agent),
        ("reddit-username", username),
        ("password", password),
    ] if not v]
    if missing:
        raise RuntimeError(f"Missing keys in .env: {missing}")

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password,
    )


_reddit: Optional[praw.Reddit] = None


def reddit_client() -> praw.Reddit:
    global _reddit
    if _reddit is None:
        _reddit = make_reddit_client()
    return _reddit


# =========================
# CSV + verified file I/O
# =========================
def load_working_csv(path: str) -> None:
    global _rows, _fieldnames, _sid_to_rowidx, _index_to_sid

    if not os.path.exists(path):
        raise FileNotFoundError(f"WORKING_CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header.")
        _fieldnames = list(reader.fieldnames)
        _rows = []
        _sid_to_rowidx = {}
        _index_to_sid = {}

        for row in reader:
            sid = (row.get(COL_SUBMISSION_ID) or "").strip()
            idx = (row.get(COL_INDEX) or "").strip()

            rowpos = len(_rows)
            _rows.append(row)

            if sid:
                if sid in _sid_to_rowidx:
                    raise RuntimeError(f"Duplicate SubmissionID in working CSV: {sid}")
                _sid_to_rowidx[sid] = rowpos

            if idx and sid:
                _index_to_sid[idx] = sid


def atomic_write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_verified(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "SubmissionID" in reader.fieldnames:
            for row in reader:
                sid = (row.get("SubmissionID") or "").strip()
                if sid:
                    out.add(sid)
        else:
            f.seek(0)
            for line in f:
                sid = line.strip()
                if sid and sid.lower() != "submissionid":
                    out.add(sid)
    return out


def append_verified(path: str, sid: str) -> None:
    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["SubmissionID"])
            w.writerow([sid])
        return

    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([sid])


# =========================
# Transcription helpers
# =========================
def safe_json_loads(s: str) -> Optional[dict]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def get_messages_from_row(row: Dict[str, str]) -> List[dict]:
    raw = row.get(COL_TRANSCRIPTION, "") or ""
    obj = safe_json_loads(raw)
    if not isinstance(obj, dict):
        return []
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return []
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        sp = m.get("speaker", "")
        tx = m.get("text", "")
        if not isinstance(sp, str) or not isinstance(tx, str):
            continue
        out.append({"speaker": sp, "text": tx})
    return out


def compute_word_count(messages: List[dict]) -> int:
    total = 0
    for m in messages:
        text = (m.get("text") or "")
        if not isinstance(text, str):
            continue
        total += len(_WORD_RE.findall(text))
    return total


def compute_speaker_turns(messages: List[dict]) -> int:
    last = None
    turns = 0
    for m in messages:
        text = (m.get("text") or "")
        if not isinstance(text, str) or text.strip() == "":
            continue
        sp = m.get("speaker", "")
        if not isinstance(sp, str):
            continue
        if sp != last:
            turns += 1
            last = sp
    return turns


def normalize_messages_for_save(messages: List[dict]) -> List[dict]:
    out = []
    for m in messages:
        sp = m.get("speaker")
        tx = m.get("text")
        if not isinstance(sp, str) or not isinstance(tx, str):
            continue
        out.append({"speaker": sp, "text": tx})
    return out


def set_row_messages(row: Dict[str, str], messages: List[dict]) -> None:
    messages = normalize_messages_for_save(messages)
    row[COL_TRANSCRIPTION] = json.dumps({"messages": messages}, ensure_ascii=False)


# =========================
# Image caching + fetching
# =========================
def ensure_cache_dir(sid: str) -> str:
    d = os.path.join(IMAGE_CACHE_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d


def numeric_sort_key(name: str) -> Tuple[int, str]:
    m = _NUM_IN_NAME.search(name)
    if m:
        try:
            return (int(m.group(1)), name)
        except Exception:
            pass
    return (10**9, name)


def list_cached_images(sid: str) -> List[str]:
    d = os.path.join(IMAGE_CACHE_DIR, sid)
    if not os.path.isdir(d):
        return []
    files = [n for n in os.listdir(d) if n.lower().endswith(_IMAGE_EXTS)]
    files.sort(key=numeric_sort_key)
    return files


def guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "gif" in ct:
        return ".gif"
    if "webp" in ct:
        return ".webp"
    return ".jpg"


def sanitize_url(url: str) -> str:
    return (url or "").replace("&amp;", "&")


def extract_gallery_urls_in_order(sub: praw.models.Submission) -> List[str]:
    out: List[str] = []
    gallery_data = getattr(sub, "gallery_data", None)
    media_metadata = getattr(sub, "media_metadata", None)
    if not isinstance(gallery_data, dict) or not isinstance(media_metadata, dict):
        return out
    items = gallery_data.get("items")
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        media_id = it.get("media_id")
        if not isinstance(media_id, str) or media_id not in media_metadata:
            continue
        meta = media_metadata.get(media_id)
        if not isinstance(meta, dict):
            continue
        s = meta.get("s")
        if isinstance(s, dict):
            u = s.get("u")
            if isinstance(u, str):
                out.append(sanitize_url(u))

    seen = set()
    dedup = []
    for u in out:
        if u and u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def extract_image_urls(sub: praw.models.Submission) -> List[str]:
    urls: List[str] = []

    try:
        if getattr(sub, "is_gallery", False):
            urls.extend(extract_gallery_urls_in_order(sub))
    except Exception:
        pass

    try:
        u = getattr(sub, "url", None)
        if isinstance(u, str):
            u2 = sanitize_url(u)
            if u2.lower().endswith(_IMAGE_EXTS) or "i.redd.it" in u2.lower():
                urls.append(u2)
    except Exception:
        pass

    try:
        prev = getattr(sub, "preview", None)
        if isinstance(prev, dict):
            imgs = prev.get("images")
            if isinstance(imgs, list) and imgs:
                src = imgs[0].get("source") if isinstance(imgs[0], dict) else None
                if isinstance(src, dict):
                    u = src.get("url")
                    if isinstance(u, str):
                        urls.append(sanitize_url(u))
    except Exception:
        pass

    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out[:MAX_IMAGES_PER_SUBMISSION]


def download_images_for_submission_id(sid: str) -> Tuple[bool, str]:
    if not ENABLE_PRAW_FETCH:
        return (False, "PRAW fetch disabled")

    ensure_cache_dir(sid)
    if list_cached_images(sid):
        return (True, "Cached")

    try:
        sub = reddit_client().submission(id=sid)
    except Exception as e:
        return (False, f"Failed to fetch submission: {e}")

    urls = extract_image_urls(sub)
    if not urls:
        return (False, "No image URLs found (deleted/non-image/unsupported)")

    n_ok = 0
    for i, url in enumerate(urls, start=1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
            resp.raise_for_status()
            ext = guess_ext_from_content_type(resp.headers.get("Content-Type", ""))
            fname = f"image{i}{ext}"
            path = os.path.join(IMAGE_CACHE_DIR, sid, fname)
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            n_ok += 1
            time.sleep(DOWNLOAD_DELAY_SECONDS)
        except Exception:
            continue

    if n_ok == 0:
        return (False, "Failed to download any images")
    return (True, f"Downloaded {n_ok} image(s)")


# =========================
# Navigation helpers
# =========================
def all_submission_ids() -> List[str]:
    return list(_sid_to_rowidx.keys())


def active_ids(unverified_only: bool) -> List[str]:
    ids = all_submission_ids()
    if unverified_only:
        return [sid for sid in ids if sid not in _verified]
    return ids


def progress_stats() -> Dict[str, int]:
    total = len(_sid_to_rowidx)
    verified = len(_verified.intersection(_sid_to_rowidx.keys()))
    remaining = max(0, total - verified)
    return {"total": total, "verified": verified, "remaining": remaining}


def get_row_by_sid(sid: str) -> Optional[Dict[str, str]]:
    ridx = _sid_to_rowidx.get(sid)
    if ridx is None:
        return None
    return _rows[ridx]


def build_item_payload(sid: str, unverified_only: bool) -> Dict:
    row = get_row_by_sid(sid)
    if row is None:
        return {"error": "unknown SubmissionID"}

    msg = ""
    images_available = False
    if ENABLE_PRAW_FETCH:
        ok, msg = download_images_for_submission_id(sid)
        images_available = ok

    cached = list_cached_images(sid)
    if cached:
        images_available = True

    image_urls = [f"/cache/{sid}/{name}" for name in cached]

    messages = get_messages_from_row(row)
    bad_speaker = any(m.get("speaker") not in ALLOWED_SPEAKERS for m in messages)
    turns = compute_speaker_turns(messages)
    words = compute_word_count(messages)

    return {
        "sid": sid,
        "csv_index": row.get(COL_INDEX, ""),
        "verified": (sid in _verified),
        "unverified_only": unverified_only,
        "progress": progress_stats(),
        "images": {"available": images_available, "status": msg, "urls": image_urls},
        "transcript": {"messages": messages, "turns": turns, "words": words, "speaker_ok": (not bad_speaker)},
    }


def navigate(sid: str, direction: str, unverified_only: bool) -> Optional[str]:
    ids = active_ids(unverified_only)
    if not ids:
        return None
    if sid not in ids:
        return ids[0]

    i = ids.index(sid)
    if direction == "next":
        j = min(len(ids) - 1, i + 1)
    else:
        j = max(0, i - 1)
    return ids[j]


# =========================
# Routes
# =========================
@app.route("/")
def index() -> Response:
    return Response(_HTML, mimetype="text/html")


@app.route("/cache/<sid>/<path:filename>")
def serve_cached_image(sid: str, filename: str):
    d = os.path.join(IMAGE_CACHE_DIR, sid)
    return send_from_directory(d, filename)


@app.route("/api/item", methods=["GET"])
def api_item():
    sid = (request.args.get("sid") or "").strip()
    unverified_only = (request.args.get("unverified_only") or "0").strip() in ("1", "true", "True")
    ids = active_ids(unverified_only)
    if not ids:
        return jsonify({"error": "No items available (maybe everything is verified?)", "progress": progress_stats()})
    if not sid or sid not in _sid_to_rowidx:
        sid = ids[0]
    return jsonify(build_item_payload(sid, unverified_only))


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True) or {}
    sid = (data.get("sid") or "").strip()
    messages = data.get("messages") or []
    row = get_row_by_sid(sid)
    if row is None:
        return jsonify({"ok": False, "error": "unknown SubmissionID"}), 400
    if not isinstance(messages, list):
        return jsonify({"ok": False, "error": "messages must be a list"}), 400
    set_row_messages(row, messages)
    return jsonify({"ok": True})


@app.route("/api/navigate", methods=["POST"])
def api_navigate():
    data = request.get_json(force=True) or {}
    sid = (data.get("sid") or "").strip()
    direction = (data.get("direction") or "").strip()
    unverified_only = bool(data.get("unverified_only", False))
    if direction not in ("next", "prev"):
        return jsonify({"ok": False, "error": "direction must be next or prev"}), 400
    next_sid = navigate(sid, direction, unverified_only)
    if next_sid is None:
        return jsonify({"ok": False, "error": "no items"}), 400
    return jsonify({"ok": True, "item": build_item_payload(next_sid, unverified_only)})


@app.route("/api/jump_index", methods=["POST"])
def api_jump_index():
    data = request.get_json(force=True) or {}
    idx = str(data.get("index") or "").strip()
    unverified_only = bool(data.get("unverified_only", False))
    if not idx:
        return jsonify({"ok": False, "error": "index is required"}), 400
    sid = _index_to_sid.get(idx)
    if not sid:
        return jsonify({"ok": False, "error": f"Index not found: {idx}"}), 404
    return jsonify({"ok": True, "item": build_item_payload(sid, unverified_only)})


@app.route("/api/verify", methods=["POST"])
def api_verify():
    data = request.get_json(force=True) or {}
    sid = (data.get("sid") or "").strip()
    if sid not in _sid_to_rowidx:
        return jsonify({"ok": False, "error": "unknown SubmissionID"}), 400
    if sid not in _verified:
        _verified.add(sid)
        append_verified(VERIFIED_CSV, sid)
    return jsonify({"ok": True, "progress": progress_stats(), "verified": True})


@app.route("/api/exit", methods=["POST"])
def api_exit():
    atomic_write_csv(WORKING_CSV, _rows, _fieldnames)
    return jsonify({"ok": True})


# =========================
# Embedded HTML/JS
# =========================
_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Transcript Verifier</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; }
    .topbar { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-bottom: 1px solid #ddd; position: sticky; top: 0; background: #fff; z-index: 5; flex-wrap: wrap; }
    .btn { padding: 6px 10px; border: 1px solid #ccc; border-radius: 8px; background: #f7f7f7; cursor: pointer; }
    .btn:active { transform: translateY(1px); }
    .pill { padding: 4px 8px; border-radius: 999px; border: 1px solid #ddd; font-size: 12px; }
    .dirty { color: #b45309; font-weight: 600; }
    .layout { display: grid; grid-template-columns: 1fr 1fr; height: calc(100vh - 92px); }
    .pane { overflow: auto; padding: 12px; }
    .images { display: flex; flex-direction: column; gap: 10px; }
    .thumbs { display: flex; gap: 6px; flex-wrap: wrap; }
    .thumbs img { width: 90px; height: 90px; object-fit: cover; border-radius: 10px; border: 2px solid transparent; cursor: pointer; }
    .thumbs img.sel { border-color: #111; }
    .bigimg { width: 100%; max-height: 75vh; object-fit: contain; border-radius: 12px; border: 1px solid #eee; background: #fafafa; }
    .msg { border: 1px solid #eee; border-radius: 14px; padding: 10px; margin: 8px 0; }
    .speaker { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; font-weight: 700; }
    .tagA { color: #1d4ed8; }
    .tagB { color: #047857; }
    .text { width: 100%; min-height: 52px; resize: vertical; margin-top: 8px; font-family: inherit; font-size: 14px; line-height: 1.35; padding: 8px; border-radius: 10px; border: 1px solid #ddd; }
    .rowmeta { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .small { font-size: 12px; color: #555; }
    .splitBtn { font-size: 12px; padding: 4px 8px; border-radius: 999px; border: 1px solid #ddd; background: #fff; cursor: pointer; }
    .jumpWrap { display:flex; align-items:center; gap:6px; }
    .jumpInput { width: 90px; padding: 6px 8px; border: 1px solid #ccc; border-radius: 8px; }
    .headerRow { display:flex; align-items:center; gap:8px; }
    .spacer { flex:1; }

    /* Loading overlay */
    .overlay {
      position: fixed; inset: 0;
      background: rgba(255,255,255,0.72);
      display: none; align-items: center; justify-content: center;
      z-index: 999;
      backdrop-filter: blur(2px);
    }
    .overlay .card {
      border: 1px solid #ddd; border-radius: 16px;
      padding: 14px 16px; background: #fff;
      display:flex; align-items:center; gap:10px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    .spinner {
      width: 18px; height: 18px;
      border: 3px solid #ddd;
      border-top-color: #111;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <div id="overlay" class="overlay">
    <div class="card">
      <div class="spinner"></div>
      <div>Loading…</div>
    </div>
  </div>

  <div class="topbar">
    <button class="btn" id="prevBtn">◀ Prev</button>
    <button class="btn" id="nextBtn">Next ▶</button>

    <div class="jumpWrap">
      <span class="pill">Jump Index</span>
      <input class="jumpInput" id="jumpIndex" placeholder="e.g. 120" />
      <button class="btn" id="jumpBtn">Go</button>
    </div>

    <button class="btn" id="swapAllBtn" title="Swap every message speaker A↔B for this conversation">
      Swap A↔B (Make B=OP)
    </button>

    <span class="pill" id="sidPill">SID: -</span>
    <span class="pill" id="idxPill">Index: -</span>

    <label class="pill" style="display:flex;align-items:center;gap:6px;cursor:pointer;">
      <input type="checkbox" id="unverifiedOnlyChk"/>
      Unverified only
    </label>

    <label class="pill" style="display:flex;align-items:center;gap:6px;cursor:pointer;">
      <input type="checkbox" id="verifiedChk"/>
      Verified
    </label>

    <span class="pill" id="progressPill">Verified 0 / 0</span>
    <span class="pill" id="statsPill">Turns 0 • Words 0</span>
    <span class="pill dirty" id="dirtyPill" style="display:none;">● Unsaved changes</span>

    <button class="btn" id="exitBtn" style="margin-left:auto;">Exit (Save CSV)</button>
  </div>

  <div class="layout">
    <div class="pane">
      <div class="rowmeta">
        <span class="pill" id="imgStatus">Images: -</span>
        <span class="small" id="imgDetail"></span>
      </div>

      <div class="images">
        <img id="bigImg" class="bigimg" alt="No image" style="display:none;"/>
        <div class="thumbs" id="thumbs"></div>
      </div>
    </div>

    <div class="pane">
      <div id="messages"></div>
    </div>
  </div>

<script>
const UNDO_DEPTH = """ + str(UNDO_DEPTH) + r""";

let state = {
  sid: "",
  unverifiedOnly: true,
  verified: false,
  messages: [],
  dirty: false,
  undo: []
};

function showOverlay(v) {
  document.getElementById("overlay").style.display = v ? "flex" : "none";
}

async function withLoading(fn) {
  showOverlay(true);
  try { return await fn(); }
  finally { showOverlay(false); }
}

function setDirty(v) {
  state.dirty = v;
  document.getElementById("dirtyPill").style.display = v ? "inline-block" : "none";
}

function pushUndoSnapshot() {
  const snap = JSON.stringify(state.messages);
  const last = state.undo.length ? state.undo[state.undo.length - 1] : null;
  if (last === snap) return;
  state.undo.push(snap);
  if (state.undo.length > UNDO_DEPTH) state.undo.shift();
}

function canTranscriptUndoHotkey() {
  const ae = document.activeElement;
  if (!ae) return true;
  return ae.tagName !== "TEXTAREA";
}

document.addEventListener("keydown", (e) => {
  const isUndo = (e.key === "z" || e.key === "Z") && (e.ctrlKey || e.metaKey);
  if (!isUndo) return;
  if (!canTranscriptUndoHotkey()) return; // textarea native undo
  e.preventDefault();

  if (state.undo.length === 0) return;
  const snap = state.undo.pop();
  try {
    state.messages = JSON.parse(snap);
    renderMessages();
    setDirty(true);
  } catch {}
});

async function apiGetItem(sid="") {
  const qs = new URLSearchParams();
  if (sid) qs.set("sid", sid);
  qs.set("unverified_only", state.unverifiedOnly ? "1" : "0");
  const r = await fetch("/api/item?" + qs.toString());
  return await r.json();
}

async function apiSave() {
  const payload = { sid: state.sid, messages: state.messages };
  const r = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "Save failed");
  setDirty(false);
}

async function apiNavigate(direction) {
  if (state.dirty) await apiSave();
  const payload = { sid: state.sid, direction, unverified_only: state.unverifiedOnly };
  const r = await fetch("/api/navigate", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "Navigate failed");
  loadItem(j.item);
}

async function apiJumpIndex(idx) {
  if (state.dirty) await apiSave();
  const payload = { index: idx, unverified_only: state.unverifiedOnly };
  const r = await fetch("/api/jump_index", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "Jump failed");
  loadItem(j.item);
}

async function apiVerify() {
  if (state.dirty) await apiSave();
  const r = await fetch("/api/verify", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ sid: state.sid })
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "Verify failed");
  const item = await apiGetItem(state.sid);
  loadItem(item);
}

async function apiExit() {
  if (state.dirty) await apiSave();
  const r = await fetch("/api/exit", { method: "POST" });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "Exit failed");
  alert("Saved working CSV. Safe to close this tab/window.");
}

function loadItem(item) {
  if (item.error) {
    alert(item.error);
    return;
  }

  state.sid = item.sid;
  state.messages = (item.transcript && item.transcript.messages) ? item.transcript.messages : [];
  state.verified = !!item.verified;
  state.undo = [];
  setDirty(false);

  document.getElementById("sidPill").textContent = "SID: " + item.sid;
  document.getElementById("idxPill").textContent = "Index: " + (item.csv_index || "-");

  const prog = item.progress || {verified:0,total:0,remaining:0};
  document.getElementById("progressPill").textContent =
    `Verified ${prog.verified} / ${prog.total} • Remaining ${prog.remaining}`;

  const turns = item.transcript ? item.transcript.turns : 0;
  const words = item.transcript ? item.transcript.words : 0;
  document.getElementById("statsPill").textContent = `Turns ${turns} • Words ${words}`;

  // Verified checkbox
  document.getElementById("verifiedChk").checked = state.verified;

  // Images
  const img = item.images || {};
  const imgStatus = document.getElementById("imgStatus");
  const imgDetail = document.getElementById("imgDetail");
  const bigImg = document.getElementById("bigImg");
  const thumbs = document.getElementById("thumbs");
  thumbs.innerHTML = "";
  bigImg.style.display = "none";
  bigImg.src = "";

  if (!img.available || !img.urls || img.urls.length === 0) {
    imgStatus.textContent = "Images: unavailable";
    imgStatus.style.borderColor = "#ef4444";
    imgDetail.textContent = img.status || "";
  } else {
    imgStatus.textContent = "Images: available (" + img.urls.length + ")";
    imgStatus.style.borderColor = "#ddd";
    imgDetail.textContent = img.status || "";

    function select(i) {
      bigImg.src = img.urls[i];
      bigImg.style.display = "block";
      [...thumbs.querySelectorAll("img")].forEach((el, idx) => {
        el.classList.toggle("sel", idx === i);
      });
    }
    img.urls.forEach((u, i) => {
      const t = document.createElement("img");
      t.src = u;
      t.alt = "img" + (i+1);
      t.onclick = () => select(i);
      thumbs.appendChild(t);
    });
    select(0);
  }

  renderMessages();
}

function otherSpeaker(sp) {
  return (sp === "Speaker A") ? "Speaker B" : "Speaker A";
}
function speakerClass(sp) {
  return (sp === "Speaker A") ? "tagA" : "tagB";
}

function insertMessage(atIndex, speaker) {
  pushUndoSnapshot();
  const msg = { speaker: speaker || "Speaker B", text: "" };
  state.messages.splice(atIndex, 0, msg);
  setDirty(true);
  renderMessages();
  setTimeout(() => {
    const ta = document.getElementById("ta_" + atIndex);
    if (ta) ta.focus();
  }, 0);
}

function renderMessages() {
  const root = document.getElementById("messages");
  root.innerHTML = "";

  if (!Array.isArray(state.messages) || state.messages.length === 0) {
    const d = document.createElement("div");
    d.className = "pill";
    d.textContent = "No messages found in Transcription JSON.";
    root.appendChild(d);
    return;
  }

  state.messages.forEach((m, idx) => {
    const box = document.createElement("div");
    box.className = "msg";

    const header = document.createElement("div");
    header.className = "headerRow";

    const sp = document.createElement("div");
    sp.className = "speaker " + speakerClass(m.speaker);
    sp.textContent = m.speaker || "(missing speaker)";
    sp.title = "Click to toggle speaker";
    sp.onclick = () => {
      pushUndoSnapshot();
      state.messages[idx].speaker = otherSpeaker(state.messages[idx].speaker);
      setDirty(true);
      renderMessages();
    };

    const addAbove = document.createElement("button");
    addAbove.className = "splitBtn";
    addAbove.textContent = "+ Above";
    addAbove.title = "Insert a new message above this one";
    addAbove.onclick = () => insertMessage(idx, state.messages[idx].speaker);

    const addBelow = document.createElement("button");
    addBelow.className = "splitBtn";
    addBelow.textContent = "+ Below";
    addBelow.title = "Insert a new message below this one";
    addBelow.onclick = () => insertMessage(idx + 1, state.messages[idx].speaker);
    
    const delBtn = document.createElement("button");
    delBtn.className = "splitBtn";
    delBtn.textContent = "🗑️";
    delBtn.title = "Delete this message";
    delBtn.onclick = () => {
      const ok = confirm("Delete this message? (Undo: Ctrl/Cmd+Z)");
      if (!ok) return;

      pushUndoSnapshot();
      if (state.messages.length <= 1) {
        alert("Can't delete the last remaining message.");
        return;
      }
      state.messages.splice(idx, 1);
      setDirty(true);
      renderMessages();
    };


    const splitBtn = document.createElement("button");
    splitBtn.className = "splitBtn";
    splitBtn.textContent = "✂ Split here (swap)";
    splitBtn.title = "Split this message at caret position and swap speaker for the remainder";
    splitBtn.onclick = () => {
      const ta = document.getElementById("ta_" + idx);
      if (!ta) return;
      const pos = ta.selectionStart;
      const text = ta.value || "";
      if (typeof pos !== "number" || pos <= 0 || pos >= text.length) return;

      const left = text.slice(0, pos);
      const right = text.slice(pos);

      pushUndoSnapshot();
      const sp0 = state.messages[idx].speaker;
      state.messages[idx].text = left;
      state.messages.splice(idx + 1, 0, { speaker: otherSpeaker(sp0), text: right });

      setDirty(true);
      renderMessages();

      setTimeout(() => {
        const nta = document.getElementById("ta_" + (idx + 1));
        if (nta) {
          nta.focus();
          nta.selectionStart = 0;
          nta.selectionEnd = 0;
        }
      }, 0);
    };

    header.appendChild(sp);
    header.appendChild(addAbove);
    header.appendChild(addBelow);
	header.appendChild(delBtn);
    header.appendChild(splitBtn);

    const ta = document.createElement("textarea");
    ta.className = "text";
    ta.id = "ta_" + idx;
    ta.value = (m.text ?? "");
    ta.oninput = () => {
      state.messages[idx].text = ta.value;
      setDirty(true);
    };

    box.appendChild(header);
    box.appendChild(ta);
    root.appendChild(box);
  });
}

// Buttons
document.getElementById("prevBtn").onclick = () => withLoading(() => apiNavigate("prev")).catch(e => alert(e));
document.getElementById("nextBtn").onclick = () => withLoading(() => apiNavigate("next")).catch(e => alert(e));

document.getElementById("jumpBtn").onclick = () => withLoading(async () => {
  const val = (document.getElementById("jumpIndex").value || "").trim();
  if (!val) return;
  await apiJumpIndex(val);
}).catch(e => alert(e));

document.getElementById("jumpIndex").addEventListener("keydown", async (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    document.getElementById("jumpBtn").click();
  }
});

document.getElementById("unverifiedOnlyChk").onchange = async (e) => {
  await withLoading(async () => {
    if (state.dirty) await apiSave();
    state.unverifiedOnly = !!e.target.checked;
    const item = await apiGetItem(state.sid);
    loadItem(item);
  }).catch(err => alert(err));
};

document.getElementById("verifiedChk").onchange = async (e) => {
  if (!e.target.checked) {
    e.target.checked = true;
    alert("Un-verify is not supported (by design).");
    return;
  }
  const ok = confirm("Are you sure you want to mark this SubmissionID as verified?");
  if (!ok) {
    e.target.checked = false;
    return;
  }
  await withLoading(() => apiVerify()).catch(err => {
    alert(err);
    e.target.checked = state.verified;
  });
};

document.getElementById("swapAllBtn").onclick = () => {
  const ok = confirm("Swap ALL speaker labels for this conversation? (Speaker A ↔ Speaker B)");
  if (!ok) return;

  pushUndoSnapshot();
  state.messages = state.messages.map(m => ({
    speaker: otherSpeaker(m.speaker),
    text: (m.text ?? "")
  }));
  setDirty(true);
  renderMessages();
};

document.getElementById("exitBtn").onclick = () => withLoading(() => apiExit()).catch(e => alert(e));

// init
(async () => {
  state.unverifiedOnly = """ + ("true" if DEFAULT_UNVERIFIED_ONLY else "false") + r""";
  document.getElementById("unverifiedOnlyChk").checked = state.unverifiedOnly;
  await withLoading(async () => {
    const item = await apiGetItem("");
    loadItem(item);
  }).catch(e => alert(e));
})();
</script>
</body>
</html>
"""


# =========================
# Entry
# =========================
def main() -> None:
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    load_working_csv(WORKING_CSV)

    global _verified
    _verified = load_verified(VERIFIED_CSV)

    print("Loaded rows:", len(_rows))
    print("Loaded SubmissionIDs:", len(_sid_to_rowidx))
    print("Verified IDs:", len(_verified))
    print(f"Open: http://{SERVER_HOST}:{SERVER_PORT}")

    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG, threaded=True)


if __name__ == "__main__":
    main()
