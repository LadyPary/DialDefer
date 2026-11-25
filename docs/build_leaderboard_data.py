"""Regenerate docs/static/js/leaderboard-data.js from the per-model analysis CSVs.

Usage (from repo root):  python docs/build_leaderboard_data.py

The nine benchmark domains come from each model's
results/benchmark/<dir>/*analysis_summary_with_aio.csv. The r/AIO row comes from
results/aio/analysis/b judges a (main)/, which is the source of the r/AIO numbers
in the published paper. To add a model, add an entry to MODELS below and rerun.
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# results dir -> display metadata. `paper: False` marks models that are not
# part of the ACL 2026 paper's five-model analysis. `aio` is the file stem in
# results/aio/analysis/b judges a (main)/. `overall_sig` is the paper's Table 2
# significance marker (McNemar, p < .0001); None = not reported in the paper.
# For models with paper=False it is filled from docs/significance.json, which
# docs/compute_significance.py computes with the same test (and checks against
# the paper's markers for the paper's models).
MODELS = {
    "gpt_4o":               {"name": "GPT-4o",                "org": "OpenAI",  "open": False, "paper": True,  "note": "gpt-4o-2024-11-20", "aio": "aio_gpt-4o-2024-11-20_speaker_b_judges_speaker_a", "overall_sig": False},
    "gpt_4o_mini":          {"name": "GPT-4o-mini",           "org": "OpenAI",  "open": False, "paper": True,  "note": "gpt-4o-mini", "aio": "aio_gpt-4o-mini_speaker_b_judges_speaker_a", "overall_sig": True},
    "gemma-3-12b-it":       {"name": "Gemma-3-12B",           "org": "Google",  "open": True,  "paper": True,  "note": "gemma-3-12b-it", "aio": "aio_gemma-3-12b-it_speaker_b_judges_speaker_a", "overall_sig": True},
    "qwen_2.5_7b_instruct": {"name": "Qwen-2.5-7B",           "org": "Alibaba", "open": True,  "paper": True,  "note": "qwen-2.5-7b-instruct", "aio": "aio_qwen-2.5-7b-instruct_speaker_b_judges_speaker_a", "overall_sig": True},
    "gpt-5-mini":           {"name": "GPT-5-mini",            "org": "OpenAI",  "open": False, "paper": True,  "note": "gpt-5-mini", "aio": "aio_gpt-5-mini_speaker_b_judges_speaker_a", "overall_sig": True},
    "nova-lite-v1":         {"name": "Nova Lite",             "org": "Amazon",  "open": False, "paper": False, "note": "nova-lite-v1", "aio": "aio_nova-lite-v1", "overall_sig": None},
    "claude-opus-5-5":      {"name": "Claude Opus 5.5",       "org": "Anthropic", "open": False, "paper": False, "note": "claude-opus-5-5 (thinking on, default effort; see results/benchmark/claude-opus-5-5/RUN_NOTES.md)", "aio": "aio_claude-opus-5-5_speaker_b_judges_speaker_a", "overall_sig": None},
}

# Overall (10-domain) numbers as printed in the paper (Figure 3 / Table 2). The
# site shows these for the paper's models instead of recomputing the average,
# which differs by rounding in a few places (e.g. Gemma DDS 29.4 vs 29.5).
PAPER_OVERALL = {
    "gpt_4o":               {"c1": 69.6, "c2": 67.6, "dc": -2.6, "di": -1.5,  "dds": -1.1},
    "gpt-5-mini":           {"c1": 75.6, "c2": 75.4, "dc": 3.9,  "di": -4.4,  "dds": 8.3},
    "gpt_4o_mini":          {"c1": 63.6, "c2": 62.6, "dc": 3.5,  "di": -5.5,  "dds": 9.0},
    "gemma-3-12b-it":       {"c1": 62.4, "c2": 61.4, "dc": 13.8, "di": -15.7, "dds": 29.5},
    "qwen_2.5_7b_instruct": {"c1": 60.4, "c2": 59.2, "dc": 15.8, "di": -18.0, "dds": 33.8},
}

DOMAIN_KEYS = ["truthfulqa", "halueval", "plausibleqa", "gpqa", "harp",
               "amqa", "bbq", "socialiqa", "advisorqa", "aio"]


def domain_key(dataset_name):
    return dataset_name.split("_")[0]


def load_significance():
    """docs/significance.json, written by docs/compute_significance.py (optional)."""
    path = os.path.join(ROOT, "docs", "significance.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def main():
    sig_file = load_significance()
    out = []
    for d, meta in MODELS.items():
        files = glob.glob(os.path.join(ROOT, "results/benchmark", d, "*analysis_summary_with_aio.csv"))
        if len(files) != 1:
            raise SystemExit(f"{d}: expected one *analysis_summary_with_aio.csv, found {len(files)}")
        domains = {}
        aio_csv = os.path.join(ROOT, "results/aio/analysis/b judges a (main)", meta["aio"] + "_analysis.csv")
        aio_rows = list(csv.DictReader(open(aio_csv)))
        assert len(aio_rows) == 1, aio_csv
        aio_rows[0]["Dataset"] = "aio"
        bench_rows = [r for r in csv.DictReader(open(files[0])) if domain_key(r["Dataset"]) != "aio"]
        for r in bench_rows + aio_rows:
            k = domain_key(r["Dataset"])
            sig = r.get("DDS_Significant", "").strip()
            domains[k] = {
                "n": int(r["N"]),
                "dds": round(float(r["DDS"]), 2),
                "dc": round(float(r["Δ_Correct"]), 2),
                "di": round(float(r["Δ_Incorrect"]), 2),
                "c1": round(float(r["C1_Avg"]), 2),
                "c2": round(float(r["C2_Avg"]), 2),
                # r/AIO: bootstrap CI from docs/significance.json (None = not computed)
                "sig": (True if sig else False) if k != "aio" else sig_file.get(d, {}).get("aio_sig"),
            }
        missing = set(DOMAIN_KEYS) - set(domains)
        if missing:
            raise SystemExit(f"{d}: missing domains {sorted(missing)}")
        meta = {k: v for k, v in meta.items() if k != "aio"}
        # Models added after the paper: overall marker from the same pooled McNemar test.
        if not meta["paper"] and d in sig_file:
            meta["overall_sig"] = sig_file[d]["overall_sig"]
        row = {"id": d, **meta, "domains": domains}
        if d in PAPER_OVERALL:
            row["paper_overall"] = PAPER_OVERALL[d]
        out.append(row)

    js = ("// AUTO-GENERATED by docs/build_leaderboard_data.py; do not edit by hand.\n"
          "window.DIALDEFER_DATA = " + json.dumps(out, indent=1) + ";\n")
    path = os.path.join(ROOT, "docs/static/js/leaderboard-data.js")
    open(path, "w").write(js)
    print(f"wrote {path} ({len(out)} models)")


if __name__ == "__main__":
    main()
