import json
from pathlib import Path

SCHEMA_VERSION = "v1"

def load_halueval_qa(
    jsonl_path: str = "benchmarks/HaluEval.jsonl",
    split: str = "all",
):
    """
    Convert HaluEval QA subset to unified (question, correct_answers, incorrect_answers) schema.
    The file is a JSONL: one JSON object per line, each with keys:
        'knowledge', 'question', 'right_answer', 'hallucinated_answer'
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed line {i}")
                continue

            question = str(ex.get("question", "")).strip()
            right_answer = str(ex.get("right_answer", "")).strip()
            hallucinated_answer = str(ex.get("hallucinated_answer", "")).strip()
            knowledge = ex.get("knowledge", None)

            if not question or not right_answer or not hallucinated_answer:
                continue

            rec = {
                "id": f"halueval-qa-{i}",
                "dataset": "halueval_qa",
                "question": question,
                "correct_answers": [right_answer],
                "incorrect_answers": [hallucinated_answer],
                "context": knowledge,
                "meta": {
                    "split": split,
                    "schema_version": SCHEMA_VERSION,
                },
            }
            records.append(rec)
    return records


def write_halueval_qa_jsonl(
    jsonl_path: str = "benchmarks/HaluEval.jsonl",
    output_path: str = "halueval_qa_unified.jsonl",
    split: str = "all",
):
    records = load_halueval_qa(jsonl_path, split=split)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    write_halueval_qa_jsonl()
