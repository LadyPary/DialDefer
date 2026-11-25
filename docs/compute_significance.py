"""Significance markers for the leaderboard, computed with the repo's own tests.

Usage (from repo root; needs numpy/statsmodels, e.g. the project .venv):
    .venv/bin/python docs/compute_significance.py      # writes docs/significance.json
    python3 docs/build_leaderboard_data.py              # picks it up

Overall (10-domain) marker, as in the paper's Table 2 (*p < .0001, McNemar):
    bench_analyzer.run_mcnemar_test on all items pooled (9 benchmarks + r/AIO
    "B judges A", N = 3,244): C1-true vs C2-correct and C1-false vs C2-incorrect.
    A model is marked when both tests give p < .0001. For the paper's five models
    this reproduces Table 2 exactly (checked below), and it is applied unchanged
    to the models added after the paper.

r/AIO marker: the same bootstrap CI as the benchmark domains
    (bench_analyzer.calculate_bootstrap_from_records, 10,000 resamples, seed 42):
    marked when the 95% CI of DDS excludes 0.
"""
import csv
import glob
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "benchmark"))
sys.path.insert(0, os.path.join(ROOT, "docs"))
import bench_analyzer as BA  # noqa: E402
from build_leaderboard_data import MODELS, domain_key  # noqa: E402

_spec = importlib.util.spec_from_file_location("aio_analyzer", os.path.join(ROOT, "code", "aio", "aio_analyzer.py"))
AIO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AIO)

ALPHA = 1e-4
# results dir under results/aio/results/ for each model
AIO_DIR = {"gpt_4o": "gpt-4o", "gpt_4o_mini": "gpt-4o-mini", "gemma-3-12b-it": "gemma-3-12b-it",
           "qwen_2.5_7b_instruct": "qwen-2.5-7b-instruct", "gpt-5-mini": "gpt-5-mini",
           "nova-lite-v1": "nova-lite-v1", "claude-opus-5-5": "claude-opus-5-5"}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def bench_records(d):
    """Benchmark records, from the same files the leaderboard CSVs were computed from."""
    recs = []
    csv_path = glob.glob(os.path.join(ROOT, "results/benchmark", d, "*analysis_summary_with_aio.csv"))[0]
    for row in csv.DictReader(open(csv_path)):
        k = domain_key(row["Dataset"])
        if k == "aio":
            continue
        p = os.path.join(ROOT, "results/benchmark", d, row["Dataset"] + ".jsonl")
        if not os.path.exists(p):
            p = p.replace("_all.jsonl", "_fixed_all.jsonl")
        if not os.path.exists(p):
            p = [f for f in glob.glob(os.path.join(ROOT, "results/benchmark", d, f"{k}_*_all.jsonl"))
                 if domain_key(os.path.basename(f)) == k][0]
        rs = load(p)
        BA.recover_nulls_from_records(rs)
        recs += rs
    return recs


def aio_records(d):
    f = glob.glob(os.path.join(ROOT, "results/aio/results", AIO_DIR[d], "*speaker_b_judges_speaker_a_results.jsonl"))
    assert len(f) == 1, f
    rs = load(f[0])
    AIO.recover_nulls_from_records(rs)
    return rs


def main():
    out = {}
    for d, meta in MODELS.items():
        bench, aio = bench_records(d), aio_records(d)
        pooled = bench + aio
        p_corr = BA.run_mcnemar_test(pooled, "c1_true_statement_ans_t1", "c2_correct_chat_ans_t1")["p_value"]
        p_inc = BA.run_mcnemar_test(pooled, "c1_false_statement_ans_t1", "c2_incorrect_chat_ans_t1")["p_value"]
        boot = BA.calculate_bootstrap_from_records(aio)
        out[d] = {
            "n_pooled": len(pooled),
            "overall_p_correct": p_corr,
            "overall_p_incorrect": p_inc,
            "overall_sig": bool(p_corr < ALPHA and p_inc < ALPHA),
            "aio_dds": round(float(boot["dds"]), 2),
            "aio_dds_ci": [round(float(x), 2) for x in boot["dds_ci"]],
            "aio_sig": bool(boot["dds_significant"]),
        }
        r = out[d]
        print(f'{meta["name"]:16s} N={r["n_pooled"]}  p_corr={p_corr:.1e}  p_inc={p_inc:.1e}  overall*={r["overall_sig"]}  '
              f'r/AIO {r["aio_dds"]:+.1f} [{r["aio_dds_ci"][0]:+.1f}, {r["aio_dds_ci"][1]:+.1f}] *={r["aio_sig"]}')
        # The test must reproduce the paper's Table 2 markers for the paper's models.
        if meta["paper"] and r["overall_sig"] != meta["overall_sig"]:
            raise SystemExit(f'{meta["name"]}: computed overall significance {r["overall_sig"]} != paper {meta["overall_sig"]}')

    path = os.path.join(ROOT, "docs", "significance.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
