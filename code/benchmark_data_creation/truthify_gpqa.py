import csv
import json
from pathlib import Path

SCHEMA_VERSION = "v1"

def convert_gpqa_csv(input_path: str, output_path: str, dataset_name: str = "gpqa"):
    """
    Convert GPQA CSV format into unified JSONL structure.
    Uses the *revised* question and answer fields.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:
        
        reader = csv.DictReader(f_in)

        for row in reader:
            q = (row.get("Question") or "").strip()
            ca = (row.get("Correct Answer") or "").strip()

            # Skip items missing essential data
            if not q or not ca:
                continue

            incorrect = []
            for col in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
                val = row.get(col)
                if val and val.strip():
                    incorrect.append(val.strip())

            record_id = row.get("Record ID") or f"{dataset_name}-{total}"

            record = {
                "id": f"{dataset_name}-{record_id}",
                "dataset": dataset_name,
                "question": q,
                "correct_answers": [ca],
                "incorrect_answers": incorrect,
                "context": None,
                "meta": {
                    "subdomain": row.get("Subdomain"),
                    "domain": row.get("High-level domain"),
                    "writer_difficulty": row.get("Writer's Difficulty Estimate"),
                    "explanation": row.get("Explanation"),
                    "schema_version": SCHEMA_VERSION
                }
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    print(f"✅ Converted {total} GPQA items → {output_path}")


if __name__ == "__main__":
    convert_gpqa_csv(
        input_path="benchmarks/gpqa_main.csv",
        output_path="gpqa_unified.jsonl"
    )
