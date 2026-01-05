#!/usr/bin/env python3
"""
Process AIO transcripts for clean C1/C2 experimental design.

This script:
1. Identifies speakers (Speaker A/B → actual identities)
2. Relabels transcript with real identities (for C2: dialogue format)
3. Converts transcript to prose narrative (for C1: prose format)

Key assumption: Speaker B is always the OP (Person A in our naming)

Usage:
    python aio_transcript_processor.py --input aio_with_summaries.jsonl --output aio_processed.jsonl --model openai/gpt-4o-mini
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import create_client, chat
from aio_labels import get_labels, list_label_types

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---
# PROMPTS
# ---

SPEAKER_IDENTIFICATION_PROMPT = """Given the context about a situation, identify who Speaker A is in the conversation.

NOTE: Speaker B is always Person A (the main subject/OP). You only need to identify Speaker A.

<<<CONTEXT>>>
Title: {title}

{body}
<<<END_CONTEXT>>>

<<<CONVERSATION>>>
{transcript}
<<<END_CONVERSATION>>>

Based on the context, who is Speaker A? 
(e.g., "Person A's mother", "Person A's husband", "Person A's coworker", "Person A's friend", "Person A's ex")

Return ONLY a JSON object:
{{"speaker_a_identity": "Person A's [relationship]", "reasoning": "brief explanation"}}

JSON:"""


BODY_NEUTRALIZATION_PROMPT = """Rewrite this text in third person. Replace all first-person references with "{label_b}" (the person who wrote this post).

Rules:
1. Replace "I", "me", "my", "mine", "myself" → "{label_b}", "{label_b}'s", etc.
2. Replace "my husband/wife/boyfriend/girlfriend/mom/dad/friend/etc." → "{label_a}" 
3. Keep ALL the facts and details - do NOT summarize
4. Keep the emotional content
5. Remove any questions like "Am I overreacting?" or "AIO" - just state the facts
6. Remove "AIO" from anywhere - replace with situation description if needed
7. Remove any text that reveals who is asking for judgment (e.g., "tell me if I'm wrong")

Original text:
{{text}}

Third-person version:"""


PROSE_CONVERSION_PROMPT = """Summarize what happened in this conversation in 2-4 sentences. Describe the key events and how each person behaved.

<<<DIALOGUE>>>
{{labeled_transcript}}
<<<END_DIALOGUE>>>

Rules:
1. Keep {label_a} and {label_b} labels
2. Summarize the main points - do NOT quote every line
3. Describe what happened and how people acted (e.g., "{label_a} became upset and called {label_b} ungrateful")
4. Include key accusations or statements if important, but paraphrase
5. Keep it brief - this should be MUCH shorter than the original dialogue
6. Do NOT interpret motivations or add judgment about who is right/wrong

Example input:
"{label_a}: We will bring it Saturday
{label_b}: I haven't seen it
{label_a}: You're so ungrateful! We're not coming anymore."

Example output:
"{label_a} said they would bring something on Saturday. When {label_b} said they hadn't seen it, {label_a} became upset, called {label_b} ungrateful, and said they would no longer visit."

Summary:"""


# ---
# TEXT NEUTRALIZATION (first-person → third-person)
# ---

def neutralize_text(
    text: str,
    client,
    model: str,
    labels: dict = None
) -> str:
    """
    Convert first-person text to third-person using label_b for OP.
    Also removes sycophancy triggers like "AIO", "Am I", etc.
    """
    if not text:
        return text
    
    if labels is None:
        labels = get_labels("speaker")
    
    label_a = labels["A"]
    label_b = labels["B"]
    
    # Format the prompt template with labels first, then with text
    prompt_template = BODY_NEUTRALIZATION_PROMPT.format(label_a=label_a, label_b=label_b)
    prompt = prompt_template.format(text=text)
    
    try:
        neutralized, _ = chat(client, model, prompt, temperature=0.3, max_tokens=2000)
        
        # Post-processing: remove any remaining triggers the LLM missed
        # Remove "AIO" variations
        neutralized = re.sub(r'\bAIO\b', 'This situation involves', neutralized)
        neutralized = re.sub(r'\bAm I\b', f'Is {label_b}', neutralized, flags=re.IGNORECASE)
        neutralized = re.sub(r'\bam I\b', f'is {label_b}', neutralized)
        
        return neutralized
    except Exception as e:
        print(f"    Neutralization failed: {e}")
        return text


# ---
# SPEAKER IDENTIFICATION
# ---

def identify_speaker_a(
    title: str,
    body: str, 
    transcript: str,
    client,
    model: str
) -> Tuple[str, str]:
    """
    Identify who Speaker A is based on context.
    Speaker B is always Person A (the OP).
    
    Returns:
        Tuple of (speaker_a_identity, reasoning)
    """
    prompt = SPEAKER_IDENTIFICATION_PROMPT.format(
        title=title,
        body=body,
        transcript=transcript
    )
    
    try:
        reply, _ = chat(client, model, prompt, max_tokens=1000)
        
        # Handle markdown code blocks
        if "```" in reply:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', reply, re.DOTALL)
            if match:
                reply = match.group(1)
        
        data = json.loads(reply)
        return data.get("speaker_a_identity", "The other person"), data.get("reasoning", "")
    
    except Exception as e:
        print(f"    Speaker ID failed: {e}")
        # Fallback: try to infer from title
        title_lower = title.lower()
        if "husband" in title_lower or "bf" in title_lower or "boyfriend" in title_lower:
            return "Person A's partner", "inferred from title"
        elif "wife" in title_lower or "gf" in title_lower or "girlfriend" in title_lower:
            return "Person A's partner", "inferred from title"
        elif "mom" in title_lower or "mum" in title_lower or "mother" in title_lower:
            return "Person A's mother", "inferred from title"
        elif "dad" in title_lower or "father" in title_lower:
            return "Person A's father", "inferred from title"
        elif "friend" in title_lower:
            return "Person A's friend", "inferred from title"
        elif "ex" in title_lower:
            return "Person A's ex", "inferred from title"
        elif "coworker" in title_lower or "boss" in title_lower:
            return "Person A's coworker", "inferred from title"
        else:
            return "The other person", "could not identify"


# ---
# TRANSCRIPT RELABELING
# ---

def parse_transcript(transcript: str) -> list:
    """
    Parse transcript from JSON or text format.
    
    Returns list of (speaker, text) tuples.
    """
    if not transcript:
        return []
    
    # Try JSON format first
    try:
        data = json.loads(transcript)
        if isinstance(data, dict) and "messages" in data:
            return [(m.get("speaker", "Unknown"), m.get("text", "")) 
                    for m in data["messages"]]
        elif isinstance(data, list):
            return [(m.get("speaker", "Unknown"), m.get("text", "")) 
                    for m in data]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try text format: "Speaker A: text\nSpeaker B: text"
    messages = []
    for line in transcript.split('\n'):
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                if speaker and text:
                    messages.append((speaker, text))
    
    return messages


def relabel_transcript(transcript: str, speaker_a_identity: str = None, labels: dict = None) -> str:
    """
    Parse transcript and return as clean dialogue text.
    
    Args:
        transcript: Raw transcript (JSON or text format)
        speaker_a_identity: Unused - kept for API compatibility
        labels: Label config dict with 'A' and 'B' keys
    
    Returns:
        Clean dialogue format with configured labels:
        "{label_a}: message
         {label_b}: message"
    """
    messages = parse_transcript(transcript)
    
    if not messages:
        return transcript  # Return as-is if parsing failed
    
    if labels is None:
        labels = get_labels("speaker")
    
    label_a = labels["A"]
    label_b = labels["B"]
    
    # Format with configured labels
    lines = []
    for speaker, text in messages:
        # Map Speaker A/B to configured labels
        if speaker == "Speaker A":
            speaker = label_a
        elif speaker == "Speaker B":
            speaker = label_b
        lines.append(f"{speaker}: {text}")
    
    return "\n".join(lines)


# ---
# PROSE CONVERSION
# ---

def convert_to_prose(
    labeled_transcript: str,
    client,
    model: str,
    labels: dict = None
) -> str:
    """
    Convert labeled dialogue to prose narrative.
    Same content, different format.
    """
    if labels is None:
        labels = get_labels("speaker")
    
    label_a = labels["A"]
    label_b = labels["B"]
    
    # Format the prompt template with labels first, then with transcript
    prompt_template = PROSE_CONVERSION_PROMPT.format(label_a=label_a, label_b=label_b)
    prompt = prompt_template.format(labeled_transcript=labeled_transcript)
    
    try:
        prose, _ = chat(client, model, prompt, temperature=0.3, max_tokens=500)
        return prose
    except Exception as e:
        print(f"    Prose conversion failed: {e}")
        return ""


# ---
# MAIN PROCESSING
# ---

def process_record(
    record: Dict[str, Any],
    client,
    model: str,
    labels: dict = None
) -> Dict[str, Any]:
    """
    Process a single record:
    1. Neutralize title and body (first-person → third-person)
    2. Relabel transcript (with configured labels)
    3. Convert to prose narrative
    """
    if labels is None:
        labels = get_labels("speaker")
    
    result = record.copy()
    
    title = record.get("Title", "")
    body = record.get("Body", "")
    transcript = record.get("transcription_formatted", "") or record.get("Transcription", "")
    
    # Step 1: Neutralize title and body
    combined_context = f"{title}\n\n{body}" if title and body else (title or body)
    if combined_context:
        neutralized_context = neutralize_text(combined_context, client, model, labels)
        result["neutralized_context"] = neutralized_context
    else:
        result["neutralized_context"] = ""
    
    if not transcript or transcript == "[No conversation available]":
        result["speaker_a_identity"] = None
        result["speaker_identification_reasoning"] = "no transcript"
        result["transcript_dialogue"] = None
        result["transcript_prose"] = None
        return result
    
    # Step 2: Identify Speaker A (for metadata, not used in labels)
    speaker_a_identity, reasoning = identify_speaker_a(
        title, body, transcript, client, model
    )
    result["speaker_a_identity"] = speaker_a_identity
    result["speaker_identification_reasoning"] = reasoning
    
    # Step 3: Relabel transcript with configured labels
    labeled_transcript = relabel_transcript(transcript, speaker_a_identity, labels)
    result["transcript_dialogue"] = labeled_transcript
    
    # Step 4: Convert to prose (for C1 - narrative format)
    prose_narrative = convert_to_prose(labeled_transcript, client, model, labels)
    result["transcript_prose"] = prose_narrative
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process AIO transcripts for C1/C2 experiment"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--model", "-m", default="openai/gpt-4o-mini", help="Model for LLM calls")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output exists")
    parser.add_argument("--label-type", "-l", default="speaker", 
                        choices=list_label_types(),
                        help=f"Label type for speakers. Choices: {list_label_types()}")
    
    args = parser.parse_args()
    
    # Get label configuration
    labels = get_labels(args.label_type)
    print(f"Using labels: {labels['A']} / {labels['B']}")
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    client = create_client(api_key)
    
    # Load input
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if args.max_rows:
        records = records[:args.max_rows]
    
    print(f"Processing {len(records)} records...")
    
    # Process records
    results = []
    iterator = tqdm(records, desc="Processing") if HAS_TQDM else records
    
    for i, record in enumerate(iterator):
        if not HAS_TQDM:
            print(f"Processing {i+1}/{len(records)}: {record.get('SubmissionID', '?')}")
        
        result = process_record(record, client, args.model, labels)
        # Store label type in output for downstream use
        result["label_type"] = args.label_type
        results.append(result)
    
    # Save output
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nSaved {len(results)} records to {args.output}")
    
    # Summary
    with_dialogue = sum(1 for r in results if r.get("transcript_dialogue"))
    with_prose = sum(1 for r in results if r.get("transcript_prose"))
    print(f"  Records with dialogue: {with_dialogue}")
    print(f"  Records with prose: {with_prose}")
    print(f"  Label type: {args.label_type} ({labels['A']} / {labels['B']})")


if __name__ == "__main__":
    main()
