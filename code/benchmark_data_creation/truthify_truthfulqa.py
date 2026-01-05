# truthify_truthfulqa.py
import json
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "v1"


def load_truthfulqa(
    csv_path: str = "benchmarks/TruthfulQA.csv",
    split: str = "all",
):
    df = pd.read_csv(csv_path)

    records = []
    for i, row in df.iterrows():
        question = str(row["Question"]).strip()

        # Correct answers
        correct_raw = "" if pd.isna(row.get("Correct Answers")) else str(row["Correct Answers"])
        correct_answers = [a.strip() for a in correct_raw.split(";") if a.strip()]

        # Ensure Best Answer is included and first
        best_answer = str(row.get("Best Answer", "")).strip()
        if best_answer:
            if best_answer in correct_answers:
                correct_answers.remove(best_answer)
            correct_answers.insert(0, best_answer)

        # Incorrect answers
        incorrect_raw = "" if pd.isna(row.get("Incorrect Answers")) else str(row["Incorrect Answers"])
        incorrect_answers = [a.strip() for a in incorrect_raw.split(";") if a.strip()]

        # Skip degenerate rows
        if not question or not correct_answers or not incorrect_answers:
            continue

        rec = {
            "id": f"truthfulqa-{i}",
            "dataset": "truthfulqa",
            "question": question,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "context": None,
            "meta": {
                "split": split,
                "type": row.get("Type"),
                "category": row.get("Category"),
                "source": row.get("Source"),
                "row_index": int(i),
                "schema_version": SCHEMA_VERSION,
            },
        }
        records.append(rec)

    return records


def write_truthfulqa_jsonl(
    csv_path: str = "benchmarks/TruthfulQA.csv",
    output_path: str = "truthfulqa_unified.jsonl",
    split: str = "all",
):
    records = load_truthfulqa(csv_path, split=split)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    write_truthfulqa_jsonl()
