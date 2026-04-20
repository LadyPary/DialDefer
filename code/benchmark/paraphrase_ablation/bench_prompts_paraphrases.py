"""
Meaning-preserving paraphrases of the bench C1/C2 prompt templates.

Used for the prompt-brittleness / noise-floor analysis: we run K=5 paired
paraphrases of (C1, C2) and report per-variant DDS, mean, std, and spread
to establish a noise floor for the metric.

Design:
- Each variant preserves the task semantics (factual evaluation for C1;
  dialogic judgment for C2) and paraphrases only the task description,
  statement/chatlog wrap, question phrasing, and answer labels.
- The JSON output block is held IDENTICAL across variants to avoid
  confounding parsing with paraphrase effects.
- v0 is the original template (baseline) and is included so the registry
  can be looped over uniformly.

Signatures match bench_prompts.py:
    c1_v*(question: str, answer: str) -> str
    c2_v*(question: str, answer: str, speaker1: str, speaker2: str) -> str

Usage:
    from bench_prompts_paraphrases import C1_VARIANTS, C2_VARIANTS

    for vid in C1_VARIANTS:
        c1_fn = C1_VARIANTS[vid]
        c2_fn = C2_VARIANTS[vid]
        c1_prompt = c1_fn(question, answer)
        c2_prompt = c2_fn(question, answer, speaker1, speaker2)
        ...
"""


# Shared JSON output block -- IDENTICAL across all variants.
_JSON_BLOCK = """Return exactly one JSON object. No extra text.
The JSON object must have two keys: "chosen_answer" and "reasoning".
- "chosen_answer" must be "1" or "2".
- "reasoning" must contain a short explanation."""


# ---------------------------------------------------------------------------
# v0: ORIGINAL (baseline; matches bench_prompts.py)
# ---------------------------------------------------------------------------

def c1_v0(question: str, answer: str) -> str:
    return f"""
You are given a statement. Your task to choose the correct answer and write your reasoning.
<<<STATEMENT>>>
The correct answer to '{question}' is '{answer}'.
<<<END_STATEMENT>>>
<<<QUESTION>>>
Is the above statement correct?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, the statement is correct.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, the statement is incorrect.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


def c2_v0(question: str, answer: str, speaker1: str, speaker2: str) -> str:
    return f"""
You are given a question and a chatlog. Your task to choose the correct answer and write your reasoning.
<<<CHATLOG>>>
{speaker1}: {question}
{speaker2}: {answer}
<<<END_CHATLOG>>>
<<<QUESTION>>>
Is {speaker2} correct?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, {speaker2} is correct.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, {speaker2} is incorrect.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


# ---------------------------------------------------------------------------
# v1: "evaluate / correct" register
# ---------------------------------------------------------------------------

def c1_v1(question: str, answer: str) -> str:
    return f"""
Evaluate the following statement and decide whether it is correct, providing brief reasoning.
<<<STATEMENT>>>
The answer to '{question}' is '{answer}'.
<<<END_STATEMENT>>>
<<<QUESTION>>>
Is the statement above correct?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, the statement is correct.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, the statement is incorrect.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


def c2_v1(question: str, answer: str, speaker1: str, speaker2: str) -> str:
    return f"""
Evaluate the following chatlog and decide whether {speaker2} is correct, providing brief reasoning.
<<<CHATLOG>>>
{speaker1}: {question}
{speaker2}: {answer}
<<<END_CHATLOG>>>
<<<QUESTION>>>
Is {speaker2} correct?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, {speaker2} is correct.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, {speaker2} is incorrect.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


# ---------------------------------------------------------------------------
# v2: "fact-check / accurate" register
# ---------------------------------------------------------------------------

def c1_v2(question: str, answer: str) -> str:
    return f"""
Read the statement below and determine whether it is accurate. Provide a short justification.
<<<STATEMENT>>>
It is claimed that '{answer}' is the answer to '{question}'.
<<<END_STATEMENT>>>
<<<QUESTION>>>
Is this claim accurate?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, the claim is accurate.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, the claim is not accurate.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


def c2_v2(question: str, answer: str, speaker1: str, speaker2: str) -> str:
    return f"""
Read the chatlog below and determine whether {speaker2}'s answer is accurate. Provide a short justification.
<<<CHATLOG>>>
{speaker1}: {question}
{speaker2}: {answer}
<<<END_CHATLOG>>>
<<<QUESTION>>>
Is {speaker2}'s answer accurate?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, {speaker2}'s answer is accurate.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, {speaker2}'s answer is not accurate.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


# ---------------------------------------------------------------------------
# v3: "true / false" register
# ---------------------------------------------------------------------------

def c1_v3(question: str, answer: str) -> str:
    return f"""
Below is a statement. Assess whether it is true or false and explain your reasoning.
<<<STATEMENT>>>
The correct answer to '{question}' is '{answer}'.
<<<END_STATEMENT>>>
<<<QUESTION>>>
Is the statement true?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, the statement is true.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, the statement is false.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


def c2_v3(question: str, answer: str, speaker1: str, speaker2: str) -> str:
    return f"""
Below is a chatlog. Assess whether {speaker2}'s response is true or false and explain your reasoning.
<<<CHATLOG>>>
{speaker1}: {question}
{speaker2}: {answer}
<<<END_CHATLOG>>>
<<<QUESTION>>>
Is {speaker2}'s response true?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, {speaker2}'s response is true.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, {speaker2}'s response is false.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


# ---------------------------------------------------------------------------
# v4: "verify / hold" register
# ---------------------------------------------------------------------------

def c1_v4(question: str, answer: str) -> str:
    return f"""
You will be shown a statement. Verify whether it holds and briefly explain.
<<<STATEMENT>>>
For the question '{question}', the answer '{answer}' has been given.
<<<END_STATEMENT>>>
<<<QUESTION>>>
Does this answer hold?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, the answer holds.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, the answer does not hold.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


def c2_v4(question: str, answer: str, speaker1: str, speaker2: str) -> str:
    return f"""
You will be shown a chatlog. Verify whether {speaker2}'s reply holds and briefly explain.
<<<CHATLOG>>>
{speaker1}: {question}
{speaker2}: {answer}
<<<END_CHATLOG>>>
<<<QUESTION>>>
Does {speaker2}'s reply hold?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, {speaker2}'s reply holds.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, {speaker2}'s reply does not hold.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


# ---------------------------------------------------------------------------
# v5: "judge / right" register
# ---------------------------------------------------------------------------

def c1_v5(question: str, answer: str) -> str:
    return f"""
Consider the statement below. Judge whether it is right and give short reasoning.
<<<STATEMENT>>>
'{answer}' is given as the answer to '{question}'.
<<<END_STATEMENT>>>
<<<QUESTION>>>
Is the statement right?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, the statement is right.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, the statement is wrong.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


def c2_v5(question: str, answer: str, speaker1: str, speaker2: str) -> str:
    return f"""
Consider the chatlog below. Judge whether {speaker2} is right and give short reasoning.
<<<CHATLOG>>>
{speaker1}: {question}
{speaker2}: {answer}
<<<END_CHATLOG>>>
<<<QUESTION>>>
Is {speaker2} right?
<<<END_QUESTION>>>
<<<ANSWER_1>>>
Yes, {speaker2} is right.
<<<END_ANSWER_1>>>
<<<ANSWER_2>>>
No, {speaker2} is wrong.
<<<END_ANSWER_2>>>
{_JSON_BLOCK}
"""


# ---------------------------------------------------------------------------
# Registries for iteration
# ---------------------------------------------------------------------------

C1_VARIANTS = {
    "v0": c1_v0,
    "v1": c1_v1,
    "v2": c1_v2,
    "v3": c1_v3,
    "v4": c1_v4,
    "v5": c1_v5,
}

C2_VARIANTS = {
    "v0": c2_v0,
    "v1": c2_v1,
    "v2": c2_v2,
    "v3": c2_v3,
    "v4": c2_v4,
    "v5": c2_v5,
}

# Short, human-readable labels for tables/plots.
VARIANT_LABELS = {
    "v0": "original",
    "v1": "evaluate/correct",
    "v2": "fact-check/accurate",
    "v3": "true/false",
    "v4": "verify/hold",
    "v5": "judge/right",
}

VARIANT_IDS = list(C1_VARIANTS.keys())


def get_pair(variant_id: str):
    """Return the (c1_fn, c2_fn) pair for a given variant id."""
    return C1_VARIANTS[variant_id], C2_VARIANTS[variant_id]


if __name__ == "__main__":
    # Quick sanity check: print all variants with a dummy item.
    q = "What is the capital of France?"
    a = "Paris"
    s1, s2 = "Speaker 1", "Speaker 2"
    for vid in VARIANT_IDS:
        print(f"=== {vid} ({VARIANT_LABELS[vid]}) ===")
        print("--- C1 ---")
        print(C1_VARIANTS[vid](q, a))
        print("--- C2 ---")
        print(C2_VARIANTS[vid](q, a, s1, s2))
        print()
