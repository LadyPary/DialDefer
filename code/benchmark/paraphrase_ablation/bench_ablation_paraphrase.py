#!/usr/bin/env python3
"""
Paraphrase ablation for the prompt-brittleness / noise-floor analysis.

Purpose
-------
Run meaning-preserving paraphrases of the C1/C2 prompt templates (v1-v5)
on a dataset, then aggregate per-variant DDS into a spread summary to
establish a noise floor for the metric. This directly addresses the
prompt-brittleness concern raised by Reviewer DHTG and the AC, and
follows the multi-prompt evaluation methodology of Sclar et al. (ICLR 2024)
and Mizrahi et al. (TACL 2024).

This file is a separate wrapper around the main bench experiment; it does
not modify bench_run_experiment.py or bench_analyzer.py. The v0 (original)
baseline is NOT re-run here -- point the aggregator at your existing v0
summary to include it in the spread statistics.

Workflow
--------
1. Run paraphrases (outputs flat files, one per variant):
    python bench_ablation_paraphrase.py run \\
        --input data/truthfulqa.jsonl \\
        --model openai/gpt-5-mini \\
        --output-dir results/paraphrase_ablation/gpt-5-mini \\
        --variants v1 v3 v5

   Produces files named:
     {dataset}_{model_short}_ablation_para_{variant}.jsonl
   e.g. truthfulqa_gpt-5-mini_ablation_para_v1.jsonl

2. Aggregate variants directly from JSONL files (no need to run
   bench_analyzer.py separately for aggregation; the aggregator computes
   DDS inline). Optionally include the v0 baseline JSONL to get the full
   spread including the original template:
    python bench_ablation_paraphrase.py aggregate \\
        --input-dir results/paraphrase_ablation/gpt-5-mini \\
        --baseline-file results/baseline/gpt-5-mini/truthfulqa_..._all.jsonl \\
        --output results/paraphrase_ablation/gpt-5-mini_spread.csv

   You can still run bench_analyzer.py on individual variant JSONLs
   separately if you want the full per-variant report (bootstrap CIs etc).
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import create_client, chat, load_jsonl, append_jsonl, extract_json_from_response
from bench_prompts_paraphrases import (
    C1_VARIANTS, C2_VARIANTS, VARIANT_LABELS
)


# Variants available for the ablation. v0 is the baseline and is handled
# separately by the aggregator (pointed at an existing summary).
ABLATION_VARIANTS = ["v1", "v2", "v3", "v4", "v5"]


# =============================================================================
# RUNNER
# =============================================================================

def run_variant_on_record(
    record: dict,
    client,
    model: str,
    c1_fn,
    c2_fn,
    speaker1_label: str,
    speaker2_label: str,
    temperature: float,
    max_tokens: int,
    system_prompt: Optional[str],
) -> dict:
    """
    Run all four conditions (C1-true, C1-false, C2-correct, C2-incorrect)
    on a single record using the given C1/C2 paraphrase functions.

    Output field names match bench_run_experiment.py so that the produced
    JSONL is directly consumable by bench_analyzer.py.
    """
    result = record.copy()

    question = record.get("question", "")
    correct_answer = record.get("chosen_correct_answer", "")
    incorrect_answer = record.get("chosen_incorrect_answer", "")

    result["speaker1"] = speaker1_label
    result["speaker2"] = speaker2_label
    result["system_prompt"] = system_prompt

    conditions = [
        ("c1_true", c1_fn(question, correct_answer),
         "c1_true_statement_ans_t1", "c1_true_statement_reasoning_t1", "c1_true_statement_t1_history"),
        ("c1_false", c1_fn(question, incorrect_answer),
         "c1_false_statement_ans_t1", "c1_false_statement_reasoning_t1", "c1_false_statement_t1_history"),
        ("c2_correct", c2_fn(question, correct_answer, speaker1_label, speaker2_label),
         "c2_correct_chat_ans_t1", "c2_correct_chat_reasoning_t1", "c2_correct_chat_t1_history"),
        ("c2_incorrect", c2_fn(question, incorrect_answer, speaker1_label, speaker2_label),
         "c2_incorrect_chat_ans_t1", "c2_incorrect_chat_reasoning_t1", "c2_incorrect_chat_t1_history"),
    ]

    for cond_name, prompt, ans_key, reasoning_key, history_key in conditions:
        history = []
        try:
            reply, history = chat(
                client, model, prompt,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt,
            )
            response = extract_json_from_response(reply)
        except Exception as e:
            print(f"[{cond_name}] Error on {record.get('id', '?')}: {e}")
            response = {}
        result[ans_key] = response.get("chosen_answer")
        result[reasoning_key] = response.get("reasoning")
        result[history_key] = str(history)

    return result


def run_variant(
    variant_id: str,
    input_file: str,
    output_file: str,
    client,
    model: str,
    speaker1_label: str,
    speaker2_label: str,
    temperature: float,
    max_tokens: int,
    max_rows: Optional[int],
    system_prompt: Optional[str],
) -> None:
    """Run a single paraphrase variant on the full dataset."""
    c1_fn = C1_VARIANTS[variant_id]
    c2_fn = C2_VARIANTS[variant_id]

    records = load_jsonl(input_file)
    if max_rows is not None:
        records = records[:max_rows]

    print("\n" + "=" * 70)
    print(f"VARIANT {variant_id} ({VARIANT_LABELS[variant_id]})")
    print("=" * 70)
    print(f"  Input:    {input_file}")
    print(f"  Output:   {output_file}")
    print(f"  Model:    {model}")
    print(f"  Rows:     {len(records)}")
    print("=" * 70)

    # Resume support: if the output file exists, skip records whose IDs are
    # already present. IDs come from the 'id' field of each JSONL record,
    # matching bench_run_experiment.py conventions.
    completed_ids = set()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if os.path.exists(output_file):
        existing = load_jsonl(output_file)
        completed_ids = {str(r.get("id")) for r in existing if r.get("id") is not None}
        print(f"  Resume: found {len(completed_ids)} completed record(s) in {output_file}")
        if completed_ids:
            print(f"  NOTE: records are skipped by ID match. "
                  f"To force a clean re-run, delete the file first.")

    iterator = tqdm(records, desc=f"{variant_id}") if HAS_TQDM else records

    skipped = 0
    for i, record in enumerate(iterator):
        rec_id = str(record.get("id")) if record.get("id") is not None else None
        if rec_id is not None and rec_id in completed_ids:
            skipped += 1
            continue
        try:
            result = run_variant_on_record(
                record=record,
                client=client,
                model=model,
                c1_fn=c1_fn,
                c2_fn=c2_fn,
                speaker1_label=speaker1_label,
                speaker2_label=speaker2_label,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            append_jsonl(result, output_file)
        except Exception as e:
            print(f"ERROR on record {i} ({record.get('id', '?')}): {e}")
            continue

        if not HAS_TQDM and (i + 1) % 25 == 0:
            print(f"  [{variant_id}] {i + 1}/{len(records)}")

    if skipped:
        print(f"  [{variant_id}] Resumed: skipped {skipped} already-completed record(s).")


def cmd_run(args: argparse.Namespace) -> None:
    """`run` subcommand: execute paraphrase variants and write per-variant JSONLs."""
    load_dotenv()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
    client = create_client(api_key)

    # Validate requested variants.
    variants = args.variants or ABLATION_VARIANTS
    unknown = [v for v in variants if v not in C1_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant id(s): {unknown}. Valid: {list(C1_VARIANTS.keys())}")
    if "v0" in variants:
        print("NOTE: v0 is the original baseline. Skip re-running and use --baseline-file at aggregation time.")

    dataset_stem = Path(args.input).stem.replace("_formatted", "").replace("_sampled", "")
    model_short = args.model.split("/")[-1]

    os.makedirs(args.output_dir, exist_ok=True)

    for vid in variants:
        out_file = Path(args.output_dir) / f"{dataset_stem}_{model_short}_ablation_para_{vid}.jsonl"
        run_variant(
            variant_id=vid,
            input_file=args.input,
            output_file=str(out_file),
            client=client,
            model=args.model,
            speaker1_label=args.speaker1,
            speaker2_label=args.speaker2,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_rows=args.max_rows,
            system_prompt=args.system_prompt,
        )

    print("\nAll variants complete.")
    print(f"Outputs in {args.output_dir}/")

    if args.aggregate:
        print("\n--aggregate is set: running aggregation now.")

        # Auto-derive the aggregate output path if not given.
        if args.aggregate_output:
            agg_output = args.aggregate_output
        else:
            od = args.output_dir.rstrip("/\\")
            agg_output = f"{od}_spread.csv"

        agg_args = argparse.Namespace(
            input_dir=args.output_dir,
            baseline_file=args.baseline_file,
            output=agg_output,
        )
        cmd_aggregate(agg_args)
    else:
        print(f"Next: python bench_ablation_paraphrase.py aggregate --input-dir {args.output_dir} ...")


# =============================================================================
# AGGREGATOR
# =============================================================================

# Pattern used to extract the variant id from an ablation filename.
# Matches names like:
#   truthfulqa_gpt-5-mini_ablation_para_v1.jsonl
#   truthfulqa_qwen-2.5-7b-instruct_ablation_para_v3_fixed.jsonl
# i.e., allows an optional suffix (e.g., "_fixed") between the variant id
# and the .jsonl extension.
_VARIANT_FROM_NAME = re.compile(r"_ablation_para_(v\d+)(?:_[a-zA-Z0-9]+)*\.jsonl$")


def _compute_dds_from_records(records: list) -> dict:
    """
    Compute per-variant DDS and accuracy metrics from raw JSONL records.

    Ground truth (matches bench_analyzer.py convention):
      - C1_True (correct statement)   -> model should answer "1"
      - C1_False (incorrect statement) -> model should answer "2"
      - C2_Correct (correct dialogue)  -> model should answer "1"
      - C2_Incorrect (incorrect dialogue) -> model should answer "2"

    Null/unparseable answers count as incorrect, matching the main analyzer's
    default behavior (without the optional null recovery step). For the full
    report with null recovery and bootstrap CIs, run bench_analyzer.py on the
    individual JSONL file.
    """
    c1_true = np.array([
        1 if str(r.get("c1_true_statement_ans_t1", "")).strip() == "1" else 0
        for r in records
    ])
    c1_false = np.array([
        1 if str(r.get("c1_false_statement_ans_t1", "")).strip() == "2" else 0
        for r in records
    ])
    c2_correct = np.array([
        1 if str(r.get("c2_correct_chat_ans_t1", "")).strip() == "1" else 0
        for r in records
    ])
    c2_incorrect = np.array([
        1 if str(r.get("c2_incorrect_chat_ans_t1", "")).strip() == "2" else 0
        for r in records
    ])

    acc_c1_true = float(np.mean(c1_true)) * 100
    acc_c1_false = float(np.mean(c1_false)) * 100
    acc_c2_correct = float(np.mean(c2_correct)) * 100
    acc_c2_incorrect = float(np.mean(c2_incorrect)) * 100

    delta_correct = acc_c2_correct - acc_c1_true
    delta_incorrect = acc_c2_incorrect - acc_c1_false
    dds = delta_correct - delta_incorrect

    return {
        "N": len(records),
        "C1_True": acc_c1_true,
        "C1_False": acc_c1_false,
        "C2_Correct": acc_c2_correct,
        "C2_Incorrect": acc_c2_incorrect,
        "C1_Avg": (acc_c1_true + acc_c1_false) / 2,
        "C2_Avg": (acc_c2_correct + acc_c2_incorrect) / 2,
        "Δ_Correct": delta_correct,
        "Δ_Incorrect": delta_incorrect,
        "DDS": dds,
    }


def _summarize_jsonl(filepath: str, variant_id: str) -> dict:
    """Load a JSONL and produce a one-row summary tagged with the variant id."""
    records = load_jsonl(filepath)
    if not records:
        raise ValueError(f"No records in {filepath}")
    # Use the dataset field from the first record, fallback to filename stem.
    dataset = records[0].get("dataset") or Path(filepath).stem
    metrics = _compute_dds_from_records(records)
    row = {"Dataset": dataset, "Variant": variant_id, **metrics}
    return row


def cmd_aggregate(args: argparse.Namespace) -> None:
    """
    `aggregate` subcommand: combine per-variant JSONLs into a spread table.

    Produces one row per dataset with:
      - Per-variant DDS (v0..v5 where available)
      - DDS_Mean, DDS_Std, DDS_Min, DDS_Max, DDS_Spread (max - min)
      - Matching stats for Δ_Correct, Δ_Incorrect, C1_Avg, C2_Avg
    """
    rows = []

    # v0 baseline (optional) — treat as a regular variant tagged "v0".
    if args.baseline_file:
        if not os.path.exists(args.baseline_file):
            raise FileNotFoundError(f"Baseline file not found: {args.baseline_file}")
        rows.append(_summarize_jsonl(args.baseline_file, "v0"))

    # Collect variant JSONLs from --input-dir.
    if args.input_dir:
        indir = Path(args.input_dir)
        if not indir.is_dir():
            raise NotADirectoryError(f"--input-dir is not a directory: {indir}")
        matches = sorted(indir.glob("*_ablation_para_v*.jsonl"))
        # Prefer *_fixed.jsonl when both raw and fixed variants are present.
        by_variant: dict = {}
        for fp in matches:
            m = _VARIANT_FROM_NAME.search(fp.name)
            if not m:
                print(f"Skipping (could not parse variant id): {fp.name}")
                continue
            vid = m.group(1)
            if vid not in C1_VARIANTS:
                print(f"Skipping (unknown variant id '{vid}'): {fp.name}")
                continue
            # If a _fixed version exists for this variant, it wins.
            prev = by_variant.get(vid)
            if prev is None:
                by_variant[vid] = fp
            else:
                if "_fixed" in fp.name and "_fixed" not in prev.name:
                    print(f"Preferring fixed file for {vid}: {fp.name}")
                    by_variant[vid] = fp
                elif "_fixed" in prev.name and "_fixed" not in fp.name:
                    pass  # keep the already-selected fixed file
                else:
                    print(f"WARNING: multiple files for {vid}, using {prev.name} and ignoring {fp.name}")

        if not by_variant:
            raise FileNotFoundError(
                f"No usable ablation files in {indir}"
            )
        for vid, fp in sorted(by_variant.items()):
            rows.append(_summarize_jsonl(str(fp), vid))

    if not rows:
        raise ValueError(
            "No data to aggregate. Provide --input-dir and/or --baseline-file."
        )

    long_df = pd.DataFrame(rows)

    # --- Save per-variant (long-form) CSV: one row per (Dataset, Variant) ---
    per_variant_cols = [
        "Dataset", "Variant", "N",
        "C1_True", "C1_False", "C2_Correct", "C2_Incorrect",
        "C1_Avg", "C2_Avg",
        "Δ_Correct", "Δ_Incorrect", "DDS",
    ]
    per_variant_df = long_df[per_variant_cols].copy()
    # Sort for readability: dataset first, then variant id (v0, v1, ..., v5).
    per_variant_df = per_variant_df.sort_values(
        by=["Dataset", "Variant"],
        key=lambda col: col if col.name == "Dataset"
        else col.map(lambda v: int(v[1:]) if isinstance(v, str) and v.startswith("v") and v[1:].isdigit() else -1),
    ).reset_index(drop=True)

    output_path = Path(args.output)
    per_variant_path = output_path.with_name(output_path.stem + "_per_variant.csv")
    out_dir = output_path.parent
    if str(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    per_variant_df.to_csv(per_variant_path, index=False)

    # Pivot DDS to wide so each variant becomes its own column.
    wide = long_df.pivot(index="Dataset", columns="Variant", values="DDS")
    wide.columns = [f"DDS_{c}" for c in wide.columns]

    dds_cols = sorted([c for c in wide.columns if c.startswith("DDS_v")])
    wide["DDS_Mean"] = wide[dds_cols].mean(axis=1)
    wide["DDS_Std"] = wide[dds_cols].std(axis=1, ddof=1)
    wide["DDS_Min"] = wide[dds_cols].min(axis=1)
    wide["DDS_Max"] = wide[dds_cols].max(axis=1)
    wide["DDS_Spread"] = wide["DDS_Max"] - wide["DDS_Min"]
    wide["N_Variants"] = wide[dds_cols].notna().sum(axis=1)

    # Also pivot secondary metrics for inspection.
    for metric in ["Δ_Correct", "Δ_Incorrect", "C1_Avg", "C2_Avg"]:
        m = long_df.pivot(index="Dataset", columns="Variant", values=metric)
        m.columns = [f"{metric}_{c}" for c in m.columns]
        wide = wide.join(m)

    wide = wide.reset_index()

    ordered = ["Dataset", "N_Variants"] + dds_cols + \
              ["DDS_Mean", "DDS_Std", "DDS_Min", "DDS_Max", "DDS_Spread"]
    others = [c for c in wide.columns if c not in ordered]
    wide = wide[ordered + others]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    wide.to_csv(args.output, index=False)

    # Console preview.
    preview_cols = ["Dataset", "N_Variants"] + dds_cols + \
                   ["DDS_Mean", "DDS_Std", "DDS_Spread"]
    preview = wide[preview_cols].copy()
    for c in preview.columns:
        if c.startswith("DDS") and c != "DDS_Std":
            preview[c] = preview[c].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "--")
        elif c == "DDS_Std":
            preview[c] = preview[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "--")

    print("\n" + "=" * 100)
    print("PARAPHRASE SPREAD SUMMARY (DDS)")
    print("=" * 100)
    print(preview.to_string(index=False))
    print("=" * 100)
    print(f"\nSpread table (wide):  {args.output}")
    print(f"Per-variant (long):   {per_variant_path}")

    if wide["DDS_Spread"].notna().any():
        mean_spread = wide["DDS_Spread"].mean()
        max_spread = wide["DDS_Spread"].max()
        print(f"\nAcross {len(wide)} dataset(s): "
              f"mean spread = {mean_spread:.1f}pp, max spread = {max_spread:.1f}pp")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paraphrase ablation runner + aggregator for the prompt-brittleness noise floor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run", help="Run paraphrase variants on a dataset.")
    p_run.add_argument("--input", "-i", required=True, help="Input JSONL file.")
    p_run.add_argument("--model", "-m", required=True, help="Model name (e.g., openai/gpt-5-mini).")
    p_run.add_argument("--output-dir", "-o", required=True,
                       help="Output directory. One flat JSONL per variant will be written, "
                            "named {dataset}_{model_short}_ablation_para_{variant}.jsonl.")
    p_run.add_argument("--variants", nargs="+", default=None,
                       help=f"Variants to run (default: {' '.join(ABLATION_VARIANTS)}).")
    p_run.add_argument("--speaker1", default="Speaker 1")
    p_run.add_argument("--speaker2", default="Speaker 2")
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--max-tokens", type=int, default=300)
    p_run.add_argument("--max-rows", type=int, default=None)
    p_run.add_argument("--api-key", default=None)
    p_run.add_argument("--system-prompt", default=None)
    p_run.add_argument("--aggregate", action="store_true",
                       help="After running all variants, automatically run the aggregate step.")
    p_run.add_argument("--baseline-file", default=None,
                       help="Optional v0 baseline JSONL path. Used by aggregation when --aggregate is set.")
    p_run.add_argument("--aggregate-output", default=None,
                       help="Optional override for the spread CSV path. "
                            "Defaults to {output-dir}_spread.csv when --aggregate is set.")
    p_run.set_defaults(func=cmd_run)

    # aggregate
    p_agg = sub.add_parser("aggregate",
                           help="Aggregate per-variant JSONLs into a DDS spread table.")
    p_agg.add_argument("--input-dir", default=None,
                       help="Directory containing flat paraphrase JSONLs "
                            "(e.g., truthfulqa_gpt-5-mini_ablation_para_v1.jsonl). "
                            "Files matching '*_ablation_para_v*.jsonl' will be loaded.")
    p_agg.add_argument("--baseline-file", default=None,
                       help="Optional path to the v0 baseline JSONL to include "
                            "alongside the paraphrase variants.")
    p_agg.add_argument("--output", "-o", required=True, help="Output CSV path for the spread table.")
    p_agg.set_defaults(func=cmd_aggregate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
