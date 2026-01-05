"""
Prompt templates for AIO (AmIOverreacting) experiment conditions.

MAIN EXPERIMENT: Dialogic Format Effect in Social Judgment
- C1: Conversation as NARRATIVE (collapsed, emotional content preserved)
- C2: Conversation as DIALOGUE (turn-by-turn, raw)
- Same information, different format
- Speaker B = OP (the one who may be overreacting)

Research question: Does seeing turn-by-turn dialogue shift judgment compared to narrative description?
"""


# ---
# C1: NARRATIVE FORMAT (collapsed description)
# ---

def c1_aio_factual_prompt(context: str, narrative: str, judgment: str, label_b: str = "Speaker B") -> str:
    """
    C1 condition: Conversation presented as narrative description.
    
    Args:
        context: Neutralized title + body
        narrative: Collapsed narrative of the conversation
        judgment: "overreacting" or "not overreacting"
        label_b: Label for the person being judged (default: "Speaker B")
    
    Returns:
        Formatted prompt string
    """
    return f"""You are given a situation and a description of a conversation that took place. Your task is to evaluate whether the statement is correct.

<<<SITUATION>>>
{context}
<<<END_SITUATION>>>

<<<CONVERSATION_DESCRIPTION>>>
{narrative}
<<<END_CONVERSATION_DESCRIPTION>>>

<<<STATEMENT>>>
{label_b} is {judgment}.
<<<END_STATEMENT>>>

<<<QUESTION>>>
Is the statement correct?
<<<END_QUESTION>>>

<<<ANSWER_1>>>
Yes, the statement is correct.
<<<END_ANSWER_1>>>

<<<ANSWER_2>>>
No, the statement is incorrect.
<<<END_ANSWER_2>>>

Return exactly one JSON object. No extra text.
The JSON object must have two keys: "chosen_answer" and "reasoning".
- "chosen_answer" must be "1" or "2".
- "reasoning" must contain a short explanation."""


# ---
# C2: DIALOGUE FORMAT (turn-by-turn)
# ---

def c2_aio_conversation_prompt(
    context: str,
    dialogue: str,
    judgment: str,
    label_b: str = "Speaker B"
) -> str:
    """
    C2 condition: Conversation presented as turn-by-turn dialogue.
    
    Same content as C1, different format.
    
    Args:
        context: Neutralized title + body
        dialogue: Raw dialogue with Speaker A/B labels
        judgment: "overreacting" or "not overreacting"
        label_b: Label for the person being judged (default: "Speaker B")
    
    Returns:
        Formatted prompt string
    """
    return f"""You are given a situation and a conversation that took place. Your task is to evaluate whether the statement is correct.

<<<SITUATION>>>
{context}
<<<END_SITUATION>>>

<<<CONVERSATION>>>
{dialogue}
<<<END_CONVERSATION>>>

<<<STATEMENT>>>
{label_b} is {judgment}.
<<<END_STATEMENT>>>

<<<QUESTION>>>
Is the statement correct?
<<<END_QUESTION>>>

<<<ANSWER_1>>>
Yes, the statement is correct.
<<<END_ANSWER_1>>>

<<<ANSWER_2>>>
No, the statement is incorrect.
<<<END_ANSWER_2>>>

Return exactly one JSON object. No extra text.
The JSON object must have two keys: "chosen_answer" and "reasoning".
- "chosen_answer" must be "1" or "2".
- "reasoning" must contain a short explanation."""


# ---
# OBSERVER LABEL OPTIONS (for ablation studies)
# ---

OBSERVER_LABELS = {
    "neutral": "An observer",
    "reddit": "A Reddit commenter", 
    "expert": "A relationship counselor",
    "ai": "An AI assistant",
    "friend": "A close friend",
    "stranger": "A stranger",
}


# ---
# HELPER: BODY NEUTRALIZATION PROMPT
# ---

def get_neutralization_prompt(body: str) -> str:
    """
    Prompt to convert first-person body text to third-person.
    
    This removes the sycophancy confound where models might side with "I".
    
    Args:
        body: Original Reddit post body (first person)
    
    Returns:
        Prompt for LLM to neutralize the text
    """
    return f"""Convert the following Reddit post into an objective third-person description.

INSTRUCTIONS:
1. Replace "I", "me", "my" with "Person A" (the one describing the situation)
2. Replace "we", "our" with "Person A and [relevant person]"
3. Refer to other people by their relationship role (e.g., "Person A's husband", "Person A's mother", "Person A's coworker")
4. Keep all facts and details the same
5. REMOVE any questions like "Am I overreacting?", "What should I do?", "Is this normal?"
6. REMOVE greetings like "Hey all", "Hi everyone", "Throwaway account because..."
7. REMOVE meta-commentary about the post itself (e.g., "I'll keep this short", "Sorry for formatting")
8. Do not add any judgment or commentary

Original post:
{body}

Objective third-person description (using "Person A" for the narrator, facts only):"""


# ---
# HELPER: FACTUAL SUMMARY GENERATION PROMPT  
# ---

def get_summary_prompt(title: str, neutralized_body: str, transcription: str) -> str:
    """
    Prompt to generate a neutral factual summary from title + body + transcription.
    
    This creates the "question equivalent" for C1 condition.
    
    Args:
        title: Post title (often contains key context)
        neutralized_body: Post context in third person
        transcription: The text conversation (JSON or formatted string)
    
    Returns:
        Prompt for LLM to generate summary
    """
    return f"""Summarize the following situation in 2-3 neutral sentences.

INSTRUCTIONS:
1. Use "Person A" for the main subject (do NOT use "the poster")
2. Use relationship roles for others (e.g., "Person A's husband", "Person A's mother")
3. Focus on: who is involved, what happened, and what the conflict is about
4. Do NOT include any judgment about whether anyone is overreacting
5. Do NOT reference "Speaker A" or "Speaker B" from the conversation - translate them to relationship roles
6. Write as an objective description of events, not from anyone's perspective

Title: {title}

Context (main situation):
{neutralized_body}

Conversation transcript (supplementary):
{transcription}

Neutral factual summary (using "Person A" for the subject):"""


# ---
# HELPER: LABEL EXTRACTION PROMPT
# ---

def get_label_extraction_prompt(comment: str) -> str:
    """
    Prompt to extract OR/NOR label from a Reddit comment.
    
    Used when pattern matching fails to find explicit OR/NOR.
    
    Args:
        comment: The Reddit comment (typically BestComment)
    
    Returns:
        Prompt for LLM to classify the comment
    """
    return f"""You are analyzing a Reddit comment from r/AmIOverreacting.
In this subreddit, people post situations and ask if they are overreacting.
Commenters typically indicate whether the poster is:
- Overreacting (OR): The poster's reaction is excessive or unjustified
- Not Overreacting (NOR): The poster's reaction is reasonable or justified

Analyze this comment and determine the commenter's stance:

Comment: "{comment}"

Return exactly one JSON object with two keys:
- "label": either "OR", "NOR", or "UNCLEAR" (if the comment doesn't take a clear stance)
- "confidence": a number from 0 to 1 indicating your confidence

JSON:"""


# ---
# COMPARISON: BENCHMARK vs AIO PROMPT STRUCTURE
# ---
"""
BENCHMARK (bench_prompts.py):
-----------------------------
C1: "The correct answer to '{question}' is '{answer}'."
    "Is the above statement correct?"
    
C2: "{speaker1}: {question}"
    "{speaker2}: {answer}"
    "Is {speaker2} correct?"

AIO (this file):
----------------
C1: "Based on this situation, the poster is {judgment}."
    "Is the above statement correct?"
    
C2: [Context + Conversation]
    "{observer} states: 'Based on this conversation, the poster is {judgment}.'"
    "Is {observer} correct?"

The parallel structure ensures:
1. Same evaluation logic (is a claim correct?)
2. Same output format (chosen_answer: 1 or 2)
3. Compatible with existing analyzer
4. Clean comparison of DDS between benchmark and AIO
"""
