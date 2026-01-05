import json
from pathlib import Path

SCHEMA_VERSION = "v1"

LETTER2IDX = {"A": 0, "B": 1, "C": 2}

def _extract_label(ex):
    """
    Returns the integer index of the correct answer (0,1,2).
    Accepts either `label_ix` (int) or `label_letter` ('A'|'B'|'C').
    """
    if "label_ix" in ex and isinstance(ex["label_ix"], int):
        li = ex["label_ix"]
        if li in (0, 1, 2):
            return li
    if "label_letter" in ex and isinstance(ex["label_letter"], str):
        letter = ex["label_letter"].strip().upper()
        if letter in LETTER2IDX:
            return LETTER2IDX[letter]
    return None

def convert_socialiqa_like(input_path: str, output_path: str, dataset_name: str = "socialiqa_like"):
    """
    Convert SocialIQA-like JSONL to unified schema:
      - question := context + " " + question
      - correct_answers := [answers[label]]
      - incorrect_answers := other two
      - context := None
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed
                continue

            context = (ex.get("context") or "").strip()
            q = (ex.get("question") or "").strip()
            aA = ex.get("answerA"); aB = ex.get("answerB"); aC = ex.get("answerC")
            answers = [aA, aB, aC]

            # Basic validation
            if not q or not any(answers):
                continue
            if any(a is None or not str(a).strip() for a in answers):
                # require all three answers present and non-empty
                continue

            label = _extract_label(ex)
            if label is None or not (0 <= label < 3):
                continue

            correct = str(answers[label]).strip()
            incorrect = [str(answers[j]).strip() for j in range(3) if j != label]
            combined_q = f"{context} {q}".strip() if context else q

            record = {
                "id": f"{dataset_name}-{i}",
                "dataset": dataset_name,
                "question": combined_q,
                "correct_answers": [correct],
                "incorrect_answers": incorrect,
                "context": None,  # merged into question per your requirement
                "meta": {
                    "schema_version": SCHEMA_VERSION,
                    # keep any useful fields if present
                    "promptDim": ex.get("promptDim"),
                    "charmap": ex.get("charmap"),
                    "answerSourcesOrigins": ex.get("answerSourcesOrigins"),
                    "answerSourcesWithCor": ex.get("answerSourcesWithCor"),
                    "promptQuestionFocusChar": ex.get("promptQuestionFocusChar"),
                },
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
            total += 1

    print(f"✅ Converted {total} examples → {out}")

if __name__ == "__main__":
    # Example usage
    convert_socialiqa_like(
        input_path="benchmarks/SocialIQA/socialIWa_v1.4_tst_wDims.jsonl",      # your source file
        output_path="socialiqa_unified.jsonl",
        dataset_name="socialiqa"
    )
