import json
from pathlib import Path

SCHEMA_VERSION = "v1"

def convert_amqa(input_path: str, output_path: str, dataset_name: str = "amqa"):
    """
    Converts AMQA (Adversarial Medical QA) dataset into unified schema.
    - Keeps original_question and desensitized_question.
    - Extracts correct and incorrect answers from options.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed line {i}")
                continue

            q_id = ex.get("question_id", i)
            question = ex.get("original_question", "").strip()
            desensitized = ex.get("desensitized_question", "").strip()
            answer = ex.get("answer", "").strip()
            options = ex.get("options", {})

            if not question or not options or not answer:
                continue

            incorrect_answers = [opt for opt in options.values() if opt.strip() != answer]

            record = {
                "id": f"{dataset_name}-{q_id}",
                "dataset": dataset_name,
                "question": question,
                "desensitized_question": desensitized,
                "correct_answers": [answer],
                "incorrect_answers": incorrect_answers,
                "context": None,
                "meta": {
                    "answer_idx": ex.get("answer_idx"),
                    "schema_version": SCHEMA_VERSION,
                    "variants": [
                        key.replace("adv_question_", "")
                        for key in ex.keys()
                        if key.startswith("adv_question_")
                    ],
                },
            }

            json.dump(record, f_out, ensure_ascii=False)
            f_out.write("\n")
            total += 1

    print(f"✅ Converted {total} items → {output_path}")


if __name__ == "__main__":
    convert_amqa(
        input_path="benchmarks/AMQA_Dataset.jsonl",
        output_path="amqa_unified.jsonl"
    )
