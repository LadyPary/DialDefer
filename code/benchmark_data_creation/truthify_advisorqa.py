import json
from pathlib import Path

SCHEMA_VERSION = "v1"

def convert_advisorqa(
    input_path: str,
    output_path: str,
    dataset_name: str = "advisorqa"
):
    """
    Convert AdvisorQA dataset into unified QA format.
    - question = first unique prefix text
    - correct_answers = suffix[sft_index]
    - incorrect_answers = other suffix entries
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed line {i}")
                continue

            prefixes = ex.get("prefix", [])
            suffixes = ex.get("suffix", [])
            reward = ex.get("reward", [])
            sft_index = ex.get("sft_index", None)
            label = ex.get("label", None)

            if not prefixes or not suffixes or sft_index is None:
                continue

            # Deduplicate prefix text and take the first
            unique_prefixes = list(dict.fromkeys(prefixes))
            question = unique_prefixes[0].strip()

            # Correct answer = SFT / preferred answer
            try:
                correct = suffixes[sft_index].strip()
            except:
                continue  # skip malformed example

            # Incorrect answers = everything else
            incorrect = []
            for j, ans in enumerate(suffixes):
                if j != sft_index:
                    ans = ans.strip()
                    if ans:
                        incorrect.append(ans)

            # Build unified record
            record = {
                "id": f"{dataset_name}-{i}",
                "dataset": dataset_name,
                "question": question,
                "correct_answers": [correct],
                "incorrect_answers": incorrect,
                "context": None,
                "meta": {
                    "reward": reward,
                    "sft_index": sft_index,
                    "label": label,  # "safe" or "unsafe"
                    "num_candidates": len(suffixes),
                    "schema_version": SCHEMA_VERSION
                }
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    print(f"✅ Converted {total} AdvisorQA entries → {output_path}")


if __name__ == "__main__":
    # Example usage:
    convert_advisorqa(
        input_path="benchmarks/AdvisorQA.jsonl",
        output_path="advisorqa_unified.jsonl"
    )
