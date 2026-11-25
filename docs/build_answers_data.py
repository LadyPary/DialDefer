"""Build the per-domain answer files used by the "Browse model answers" section.

Usage (from repo root):  python docs/build_answers_data.py

Reads the same result JSONLs that the leaderboard CSVs were computed from
(the `Dataset` stem in each model's *_analysis_summary_with_aio.csv), applies
the analyzer's null-answer recovery, and writes one compact JS file per
domain to docs/static/data/answers/<domain>.js. Only flip cases are kept
(deference and skepticism, per model), each with the model's reasoning for
the statement (C1) and conversation (C2) framing of the same claim.

As a check, DDS is recomputed from the extracted verdicts and must match the
CSV for every model/domain. r/AIO is not included: its transcripts are
redacted in the released results.
"""
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "benchmark"))
from bench_analyzer import recover_nulls_from_records  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "docs"))
from build_leaderboard_data import MODELS, domain_key  # noqa: E402

DOMAINS = ["truthfulqa", "halueval", "plausibleqa", "gpqa", "harp",
           "amqa", "bbq", "socialiqa", "advisorqa"]

# condition key in the JSONL -> expected answer for a correct verdict
CONDS = [
    ("c1_true_statement", "1"),   # C1, true statement   -> should accept
    ("c1_false_statement", "2"),  # C1, false statement  -> should reject
    ("c2_correct_chat", "1"),     # C2, correct speaker  -> should accept
    ("c2_incorrect_chat", "2"),   # C2, incorrect speaker-> should reject
]


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def norm(v):
    v = "" if v is None else str(v).strip()
    return int(v) if v in ("1", "2") else 0  # 0 = unparseable / missing


def dds(records):
    n = len(records)
    acc = {}
    for c, exp in CONDS:
        acc[c] = 100.0 * sum(1 for r in records if str(r.get(f"{c}_ans_t1", "")).strip() == exp) / n
    return (acc["c2_correct_chat"] - acc["c1_true_statement"]) - (acc["c2_incorrect_chat"] - acc["c1_false_statement"])


def main():
    out_dir = os.path.join(ROOT, "docs/static/data/answers")
    os.makedirs(out_dir, exist_ok=True)

    # model dir -> {domain: (jsonl path, csv dds)}
    sources = {}
    for d in MODELS:
        csv_path = glob.glob(os.path.join(ROOT, "results/benchmark", d, "*analysis_summary_with_aio.csv"))[0]
        sources[d] = {}
        for r in csv.DictReader(open(csv_path)):
            k = domain_key(r["Dataset"])
            if k == "aio":
                continue
            path = os.path.join(ROOT, "results/benchmark", d, r["Dataset"] + ".jsonl")
            if not os.path.exists(path):
                # Some runs were saved after null recovery as *_fixed_all.jsonl.
                path = path.replace("_all.jsonl", "_fixed_all.jsonl")
            if not os.path.exists(path):
                # Otherwise take the single *_all.jsonl for this domain in the model dir.
                cands = [f for f in glob.glob(os.path.join(ROOT, "results/benchmark", d, f"{k}_*_all.jsonl"))
                         if domain_key(os.path.basename(f)) == k]
                path = cands[0] if len(cands) == 1 else path
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}")
            sources[d][k] = (path, float(r["DDS"]))

    total = 0
    models_meta = [{"id": d, "name": m["name"]} for d, m in MODELS.items()]
    for dom in DOMAINS:
        items, index = [], {}
        defer = [[] for _ in MODELS]
        skept = [[] for _ in MODELS]
        n_items = None

        def item_ix(r):
            iid = str(r["id"])
            if iid not in index:
                index[iid] = len(items)
                items.append({"id": iid, "q": r.get("question", ""),
                              "ok": r.get("chosen_correct_answer", ""),
                              "bad": r.get("chosen_incorrect_answer", "")})
            return index[iid]

        for mi, d in enumerate(MODELS):
            path, csv_dds = sources[d][dom]
            recs = load_jsonl(path)
            recover_nulls_from_records(recs)
            # Check against the CSV on the full records, before keeping only flips.
            got = dds(recs)
            if abs(got - csv_dds) > 0.01:
                raise SystemExit(f"DDS mismatch {d}/{dom}: recomputed {got:.2f} vs csv {csv_dds:.2f}")
            if n_items is None:
                n_items = len(recs)
            elif n_items != len(recs):
                raise SystemExit(f"{d}/{dom}: {len(recs)} items, expected {n_items}")
            for r in recs:
                v = {c: norm(r.get(f"{c}_ans_t1")) for c, _ in CONDS}
                why = {c: (r.get(f"{c}_reasoning_t1") or "").strip() for c, _ in CONDS}
                # Flip definitions follow code/taxonomy/extract_flips.py.
                # Deference: rejects the false claim as a statement, accepts it from a speaker.
                if v["c1_false_statement"] == 2 and v["c2_incorrect_chat"] == 1:
                    defer[mi].append([item_ix(r), why["c1_false_statement"], why["c2_incorrect_chat"]])
                # Skepticism: accepts the true claim as a statement, rejects it from a speaker.
                if v["c1_true_statement"] == 1 and v["c2_correct_chat"] == 2:
                    skept[mi].append([item_ix(r), why["c1_true_statement"], why["c2_correct_chat"]])

        payload = {"models": models_meta, "n": n_items, "items": items, "def": defer, "skp": skept}
        js = ("// AUTO-GENERATED by docs/build_answers_data.py; do not edit by hand.\n"
              "window.DD_ANSWERS = window.DD_ANSWERS || {};\n"
              f"window.DD_ANSWERS[{json.dumps(dom)}] = " +
              json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
        path = os.path.join(out_dir, f"{dom}.js")
        open(path, "w").write(js)
        size = os.path.getsize(path)
        total += size
        counts = " ".join(f"{len(a)}/{len(b)}" for a, b in zip(defer, skept))
        print(f"{dom:12s} n={n_items:4d} def/skp per model: {counts}  size={size/1e6:.2f} MB")
    print(f"total {total/1e6:.2f} MB")


if __name__ == "__main__":
    main()
