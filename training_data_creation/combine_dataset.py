import json
import argparse
from pathlib import Path


def combine_jsonl(input_files, output_file, dedup_by_id: bool = False):
    """
    Combine multiple JSONL files into one.

    - input_files: list of paths to JSONL files
    - output_file: path to output JSONL file
    - dedup_by_id: if True, keep only the first occurrence of each 'id'
    """
    seen_ids = set()
    num_written = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for path in input_files:
            path = Path(path)
            print(f"Reading {path} ...")
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)

                    if dedup_by_id:
                        obj_id = obj.get("id")
                        if obj_id is not None:
                            if obj_id in seen_ids:
                                continue
                            seen_ids.add(obj_id)

                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    num_written += 1

    print(f"Done. Wrote {num_written} lines to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple JSONL files into one."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSONL files to combine.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--dedup_by_id",
        action="store_true",
        help="If set, deduplicate entries based on the 'id' field.",
    )

    args = parser.parse_args()
    combine_jsonl(args.inputs, args.output, dedup_by_id=args.dedup_by_id)


if __name__ == "__main__":
    main()
