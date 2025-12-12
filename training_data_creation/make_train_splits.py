#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def load_test_ids(test_file: Path) -> set:
    """
    Load all `id` values from a test JSONL file.
    """
    ids = set()
    if not test_file.exists():
        print(f"[warn] Test file not found, skipping: {test_file}")
        return ids

    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed line in {test_file}")
                continue
            ex_id = ex.get("id")
            if ex_id is not None:
                ids.add(ex_id)
    return ids


def create_train_split(all_dir: Path, test_dir: Path, out_dir: Path):
    """
    For each JSONL file in `all_dir`, remove examples whose `id` appears in the
    corresponding test file in `test_dir`, and write the remaining examples
    to `out_dir` with the same filename.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all JSONL files in the "all" directory
    for all_file in sorted(all_dir.glob("*.jsonl")):
        test_file = test_dir / all_file.name
        out_file = out_dir / all_file.name

        print(f"Processing {all_file.name}...")
        test_ids = load_test_ids(test_file)
        print(f"  Loaded {len(test_ids)} test IDs from {test_file.name}")

        kept = 0
        total = 0

        with all_file.open("r", encoding="utf-8") as fin, \
             out_file.open("w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[warn] Skipping malformed line in {all_file}")
                    continue

                total += 1
                ex_id = ex.get("id")
                if ex_id in test_ids:
                    continue  # drop test example

                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1

        print(f"  Kept {kept} / {total} examples → {out_file.name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create train splits by removing test IDs from full datasets."
    )
    parser.add_argument(
        "--all_dir",
        type=str,
        required=True,
        help="Directory with all unified datasets (9 JSONL files).",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory with test splits (same filenames, JSONL).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for train splits.",
    )
    args = parser.parse_args()

    all_dir = Path(args.all_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)

    create_train_split(all_dir, test_dir, out_dir)


if __name__ == "__main__":
    main()
