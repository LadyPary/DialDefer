import json
from pathlib import Path
from typing import List, Dict, Any, Optional

SCHEMA_VERSION = "v1"

def load_plausibleqa(
    json_path: str = "benchmarks/PlausibleQA.json",
    split: str = "test",
    plausibility_key: str = "init_plackett_luce",
    top_k_negatives: Optional[int] = None,
    min_plausibility: Optional[float] = None,
):
    """
    Convert PlausibleQA to unified (question, correct_answers, incorrect_answers) schema.

    Parameters
    ----------
    json_path : str
        Path to PlausibleQA JSON file.
    split : str
        Dataset split label.
    plausibility_key : str
        Which metric to use for plausibility sorting/filtering.
    top_k_negatives : int | None
        Keep only the K most plausible wrong answers.
    min_plausibility : float | None
        Keep only wrong answers with score >= this threshold.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []

    for ex in data:
        qid = ex.get("id")
        question = str(ex.get("question", "")).strip()
        correct = str(ex.get("answer", "")).strip()
        if not question or not correct:
            continue

        candidate_dict = ex.get("candidate_answers", {})
        if not isinstance(candidate_dict, dict):
            continue

        wrongs_with_scores = []
        for cand_text, cand_meta in candidate_dict.items():
            cand_text = cand_text.strip()
            if not cand_text or cand_text == correct:
                continue

            score = cand_meta.get(plausibility_key)
            try:
                score_val = float(score) if score is not None else 0.0
            except (TypeError, ValueError):
                score_val = 0.0

            if min_plausibility is not None and score_val < min_plausibility:
                continue

            wrongs_with_scores.append((cand_text, score_val))

        if not wrongs_with_scores:
            continue

        # sort descending by plausibility score
        wrongs_with_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k_negatives is not None:
            wrongs_with_scores = wrongs_with_scores[:top_k_negatives]

        incorrect_answers = [t for t, _ in wrongs_with_scores]

        qt = ex.get("question_type", {}) or {}
        rec = {
            "id": f"plausibleqa-{qid}",
            "dataset": "plausibleqa",
            "question": question,
            "correct_answers": [correct],
            "incorrect_answers": incorrect_answers,
            "context": None,
            "meta": {
                "split": split,
                "source_id": qid,
                "question_type_major": qt.get("major"),
                "question_type_minor": qt.get("minor"),
                "question_difficulty": ex.get("question_difficulty"),
                "answer_difficulty": ex.get("answer_difficulty"),
                "plausibility_key": plausibility_key,
                "schema_version": SCHEMA_VERSION,
            },
        }
        records.append(rec)

    return records


def write_plausibleqa_jsonl(
    json_path: str = "benchmarks/PlausibleQA.json",
    output_path: str = "plausibleqa_unified.jsonl",
    split: str = "test",
    plausibility_key: str = "init_plackett_luce",
    top_k_negatives: Optional[int] = 3,
    min_plausibility: Optional[float] = None,
):
    """Dump unified JSONL file for PlausibleQA."""
    records = load_plausibleqa(
        json_path=json_path,
        split=split,
        plausibility_key=plausibility_key,
        top_k_negatives=top_k_negatives,
        min_plausibility=min_plausibility,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Example: keep top-3 hardest negatives by `init_plackett_luce`
    write_plausibleqa_jsonl(top_k_negatives=3)
