import json
from pathlib import Path

SCHEMA_VERSION = "v1"

def convert_harp_mc(input_path: str, output_path: str, dataset_name: str = "harp_mc"):
    """
    Convert HARP multiple-choice math dataset into unified JSONL format.
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

            problem = ex.get("problem", "").strip()
            choices = ex.get("choices", {})
            answer_choice = ex.get("answer_choice")
            answer_text = ex.get("answer", "").strip()

            if not problem or not choices or not answer_choice:
                continue

            # Extract correct answer
            correct = choices.get(answer_choice)
            if correct is None:
                continue

            # Collect incorrect answers
            incorrect = [txt for key, txt in choices.items() if key != answer_choice]

            # Build unified record
            record = {
                "id": f"{dataset_name}-{ex.get('year')}-{ex.get('contest')}-{ex.get('number')}",
                "dataset": dataset_name,

                # QUESTION = problem itself
                "question": problem,

                "correct_answers": [correct],
                "incorrect_answers": incorrect,
                "context": None,

                "meta": {
                    "year": ex.get("year"),
                    "contest": ex.get("contest"),
                    "number": ex.get("number"),
                    "level": ex.get("level"),
                    "subject": ex.get("subject"),
                    "multiple_choice_only": ex.get("multiple_choice_only"),
                    "solution": ex.get("solution_1"),
                    "num_solutions": ex.get("num_solutions"),
                    "schema_version": SCHEMA_VERSION,
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    print(f"✅ Converted {total} items → {output_path}")


if __name__ == "__main__":
    convert_harp_mc(
        input_path="benchmarks/HARP_mcq.jsonl",           # your downloaded file
        output_path="harp_mcq_unified.jsonl"
    )
