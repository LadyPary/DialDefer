#!/usr/bin/env python3

import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple


# =========================
# Hardcoded configuration
# =========================
INPUT_CSV = "AIODataFullClean.csv"
OUTPUT_DIR = "cleaned"

OUTPUT_SWEET = os.path.join(OUTPUT_DIR, "clean_sweet_spot.csv")
OUTPUT_MIN = os.path.join(OUTPUT_DIR, "clean_minimum.csv")

TRANSCRIPTION_COL = "Transcription"
INDEX_COL = "Index"

ALLOWED_SPEAKERS = {"Speaker A", "Speaker B"}

# Turn/word bounds
MIN_TURNS_LO, MIN_TURNS_HI = 3, 30
MIN_WORDS_LO, MIN_WORDS_HI = 50, 1500

SWEET_TURNS_LO, SWEET_TURNS_HI = 5, 15
SWEET_WORDS_LO, SWEET_WORDS_HI = 100, 500

# Redaction threshold (proportion of "redaction tokens" among words)
REDACTION_RATIO_THRESHOLD = 0.30

# Voice transcription indicator phrases (case-insensitive substring match)
VOICE_INDICATORS = [
    "voice message",
    "voice note",
    "audio message",
    "audio note",
    "sent a voice",
    "sent a voice note",
    "sent a voice message",
    "sent an audio",
    "sent an audio message",
    "listen to my voice",
    "listened to your voice",
]

# Common English words for lightweight majority-English heuristic
ENGLISH_COMMON = {
    "the","and","to","of","a","in","is","it","you","that","for","on","with","was","as","are","this",
    "but","be","have","not","we","they","i","me","my","your","so","at","if","or","what","just","do",
    "no","yes","can","like","im","i'm","dont","don't","did","didn't","does","doesn't","because","about",
    "really","then","there","here","been","were","when","who","how","why","an","all","one","would",
    "should","could","them","their","our","us","he","she","him","her","his","hers","said","say","go",
    "going","went","get","got","make","made","know","knew","think","thought","want","wanted","feel",
    "felt","tell","told","also","still","even","maybe","okay","ok","yeah","nah","lol"
}


# =========================
# Helper functions
# =========================
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_REDACTION_PATTERNS = [
    re.compile(r"\[REDACTED\]", re.IGNORECASE),
    re.compile(r"\bREDACTED\b", re.IGNORECASE),
    re.compile(r"█+"),               # black bar blocks
    re.compile(r"\*{3,}"),           # *** style
]


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


def extract_messages(transcription_obj: dict) -> Optional[List[dict]]:
    if not isinstance(transcription_obj, dict):
        return None
    msgs = transcription_obj.get("messages", None)
    if not isinstance(msgs, list):
        return None
    for m in msgs:
        if not isinstance(m, dict):
            return None
        if "speaker" not in m or "text" not in m:
            return None
    return msgs


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def count_speaker_turns(messages: List[dict]) -> int:
    """
    Speaker turns = number of contiguous speaker blocks, ignoring empty texts.
    Example (empty ignored): A:"hi", A:"", A:"ok", B:"yo" => turns = 2
    """
    last = None
    turns = 0
    for m in messages:
        text = (m.get("text", "") or "")
        if text.strip() == "":
            continue
        sp = m.get("speaker", "")
        if sp != last:
            turns += 1
            last = sp
    return turns


def redaction_ratio(text: str) -> float:
    if not text:
        return 0.0

    words = _WORD_RE.findall(text)
    total_words = len(words)
    if total_words == 0:
        return 0.0

    redacted_hits = 0
    for pat in _REDACTION_PATTERNS:
        redacted_hits += len(pat.findall(text))
    redacted_hits += sum(1 for w in words if w.lower() == "redacted")

    return redacted_hits / total_words


def looks_like_voice_transcript(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(phrase in low for phrase in VOICE_INDICATORS)


def majority_english(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False

    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if len(tokens) < 10:
        return False

    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False

    ascii_alpha = sum(1 for c in alpha_chars if "a" <= c.lower() <= "z")
    ascii_ratio = ascii_alpha / max(1, len(alpha_chars))

    common_hits = sum(1 for t in tokens if t in ENGLISH_COMMON)
    common_ratio = common_hits / max(1, len(tokens))

    return (ascii_ratio >= 0.85) and (common_ratio >= 0.03)


def row_passes_filters(row: Dict[str, str]) -> Tuple[bool, bool]:
    """
    Returns (qualifies_minimum, qualifies_sweet).
    If neither, the row is filtered out entirely.
    """
    t = row.get(TRANSCRIPTION_COL, "")
    obj = safe_json_loads(t)
    if obj is None:
        return (False, False)

    messages = extract_messages(obj)
    if messages is None:
        return (False, False)

    # Validate speakers early + gather NON-empty texts
    texts: List[str] = []
    for m in messages:
        sp = m.get("speaker", "")
        if sp not in ALLOWED_SPEAKERS:
            return (False, False)

        txt = (m.get("text", "") or "")
        if txt.strip() != "":
            texts.append(txt)

    # Must have some content
    if not texts:
        return (False, False)

    # Turns are speaker-turn blocks, ignoring empty texts
    turns = count_speaker_turns(messages)
    if turns < MIN_TURNS_LO or turns > MIN_TURNS_HI:
        return (False, False)

    # Word bounds (over non-empty texts)
    wc = sum(count_words(x) for x in texts)
    if wc < MIN_WORDS_LO or wc > MIN_WORDS_HI:
        return (False, False)

    combined_text = " ".join(texts)

    # Redaction filter
    if redaction_ratio(combined_text) > REDACTION_RATIO_THRESHOLD:
        return (False, False)

    # Voice transcription filter
    if looks_like_voice_transcript(combined_text):
        return (False, False)

    # Majority English filter
    if not majority_english(combined_text):
        return (False, False)

    qualifies_minimum = True
    qualifies_sweet = (
        SWEET_TURNS_LO <= turns <= SWEET_TURNS_HI and
        SWEET_WORDS_LO <= wc <= SWEET_WORDS_HI
    )
    return (qualifies_minimum, qualifies_sweet)


def print_stats(label: str, kept: int, total: int) -> None:
    removed = total - kept
    pct = (kept / total * 100.0) if total > 0 else 0.0
    print(f"\n=== {label} ===")
    print(f"Rows in output: {kept}")
    print(f"Rows removed:   {removed}")
    print(f"% of input:     {pct:.2f}%")


def write_csv_with_renumber(
    path: str,
    rows: List[Dict[str, str]],
    fieldnames: List[str],
) -> None:
    """
    Writes rows to CSV using exact same header as input.
    If 'Index' exists in header, rewrite it to 1..N in output order.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()

        if INDEX_COL in fieldnames:
            for i, row in enumerate(rows, start=1):
                row_out = dict(row)
                row_out[INDEX_COL] = str(i)
                w.writerow(row_out)
        else:
            w.writerows(rows)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError("Input CSV appears to have no header/columns.")

        sweet_rows: List[Dict[str, str]] = []
        min_rows: List[Dict[str, str]] = []
        total_rows = 0

        for row in reader:
            total_rows += 1
            qualifies_min, qualifies_sweet = row_passes_filters(row)
            if qualifies_min:
                min_rows.append(row)
            if qualifies_sweet:
                sweet_rows.append(row)

    write_csv_with_renumber(OUTPUT_MIN, min_rows, fieldnames)
    write_csv_with_renumber(OUTPUT_SWEET, sweet_rows, fieldnames)

    print(f"Input CSV: {INPUT_CSV}")
    print(f"Total input rows: {total_rows}")
    print(f"Output (minimum): {OUTPUT_MIN}")
    print(f"Output (sweet):   {OUTPUT_SWEET}")

    print_stats("Minimum Requirements (3–30 turns, 50–1500 words)", kept=len(min_rows), total=total_rows)
    print_stats("Sweet Spot (5–15 turns, 100–500 words)", kept=len(sweet_rows), total=total_rows)


if __name__ == "__main__":
    main()
