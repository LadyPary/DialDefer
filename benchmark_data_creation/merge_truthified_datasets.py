import json
from pathlib import Path
from typing import List
from collections import Counter

def merge_truthified_datasets(
    input_paths: List[str],
    output_path: str = "merged_truthified_benchmarks.jsonl",
):
    """
    Merge multiple truthified JSONL datasets (same schema) into one.
    Automatically adds unique IDs if missing, and logs dataset counts.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    dataset_counts = Counter()
    merged_records = []

    def valid_json(line: str):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    for path in input_paths:
        p = Path(path)
        if not p.exists():
            print(f"[warn] File not found: {p}")
            continue

        dataset_name = p.stem.split("_unified")[0]
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                ex = valid_json(line)
                if not ex or "question" not in ex:
                    continue

                # ensure id
                if "id" not in ex or not ex["id"]:
                    ex["id"] = f"{dataset_name}-{i}"

                # ensure unique id across datasets
                while ex["id"] in seen_ids:
                    ex["id"] += "_dup"
                seen_ids.add(ex["id"])

                merged_records.append(ex)
                dataset_counts[ex.get("dataset", dataset_name)] += 1

    # Write output
    with output.open("w", encoding="utf-8") as f:
        for r in merged_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n✅ Merge complete.")
    print(f"➡️  Total records: {len(merged_records)}")
    for ds, count in dataset_counts.items():
        print(f"   - {ds}: {count}")
    print(f"\nMerged file saved to: {output.resolve()}")


if __name__ == "__main__":
    # Example usage:
    merge_truthified_datasets([
        "truthfulqa_unified.jsonl",
        "halueval_qa_unified.jsonl",
        "plausibleqa_unified.jsonl"
    ])
