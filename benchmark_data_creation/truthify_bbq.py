import json
from pathlib import Path

SCHEMA_VERSION = "v1"

def convert_bbq_all(input_dir: str, output_path: str, dataset_name: str = "bbq"):
    """
    Converts all BBQ .jsonl files (NYU-MLL structure) into one unified JSONL file.
    - Combines context + question into 'question'.
    - Uses ans0/ans1/ans2 and label for answers.
    """

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for file in sorted(input_dir.glob("*.jsonl")):
            print(f"🔹 Processing {file.name}...")
            with file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"[warn] Skipping malformed line {i} in {file.name}")
                        continue

                    # Extract fields
                    context = str(ex.get("context", "")).strip()
                    question = str(ex.get("question", "")).strip()
                    label = ex.get("label")

                    # Answers come from ans0/ans1/ans2
                    answers = [ex.get(f"ans{j}") for j in range(3) if f"ans{j}" in ex]

                    # Basic validation
                    if not question or not context or label is None or not answers:
                        continue
                    if not 0 <= label < len(answers):
                        continue

                    correct_answer = answers[label]
                    incorrect_answers = [a for j, a in enumerate(answers) if j != label]
                    combined_q = f"{context} {question}".strip()

                    record = {
                        "id": f"{dataset_name}-{file.stem}-{i}",
                        "dataset": dataset_name,
                        "question": combined_q,
                        "correct_answers": [correct_answer],
                        "incorrect_answers": incorrect_answers,
                        "context": None,  # merged into question
                        "meta": {
                            "source_file": file.name,
                            "category": ex.get("category"),
                            "context_condition": ex.get("context_condition"),
                            "question_polarity": ex.get("question_polarity"),
                            "schema_version": SCHEMA_VERSION,
                        },
                    }

                    json.dump(record, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    total_records += 1

    print(f"\n✅ Merged all BBQ files into: {output_path}")
    print(f"➡️  Total records written: {total_records}")


if __name__ == "__main__":
    convert_bbq_all(
        input_dir="benchmarks/BBQ_data",      # your data folder
        output_path="bbq_unified.jsonl"       # single merged file
    )
