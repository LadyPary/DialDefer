#!/usr/bin/env python3
"""
Generate neutralized bodies and factual summaries for AIO records.

This script:
1. Converts first-person Body text to third-person (neutralization)
2. Generates a neutral factual summary from Title + Neutralized Body + Transcription

Usage:
    python aio_summarizer.py --input aio_labeled.jsonl --output aio_with_summaries.jsonl --model openai/gpt-4o-mini
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from dotenv import load_dotenv
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import create_client, chat


# ---
# PROMPTS
# ---

NEUTRALIZATION_PROMPT = """Convert the following Reddit post into an objective third-person description.

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


SUMMARY_PROMPT = """Summarize the following situation in 2-3 neutral sentences.

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
# TRANSCRIPTION FORMATTING
# ---

def format_transcription(transcription_field: str) -> str:
    """
    Format transcription JSON into readable dialogue.
    
    Args:
        transcription_field: JSON string with messages
        
    Returns:
        Formatted dialogue string
    """
    if not transcription_field or transcription_field.strip() == "":
        return "[No conversation available]"
    
    try:
        data = json.loads(transcription_field)
        messages = data.get("messages", [])
        
        if not messages:
            return "[No conversation available]"
        
        lines = []
        for msg in messages:
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "")
            lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)
        
    except json.JSONDecodeError:
        # If not JSON, return as-is
        return transcription_field


# ---
# LLM CALLS
# ---

def call_llm(
    client,
    model: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 500
) -> str:
    """
    Make a simple LLM call using api_client.
    
    Args:
        client: OpenAI-compatible client
        model: Model name
        prompt: The prompt
        temperature: Sampling temperature
        max_tokens: Max response length
        
    Returns:
        Response text
    """
    response, _ = chat(
        client=client,
        model=model,
        user_message=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.strip()


def neutralize_body(
    body: str,
    client,
    model: str,
    temperature: float = 0.3
) -> str:
    """
    Convert first-person body text to third-person.
    
    Args:
        body: Original body text
        client: OpenAI client
        model: Model name
        temperature: Sampling temperature
        
    Returns:
        Neutralized body text
    """
    if not body or body.strip() == "":
        return ""
    
    prompt = NEUTRALIZATION_PROMPT.format(body=body)
    return call_llm(client, model, prompt, temperature=temperature, max_tokens=800)


def generate_summary(
    title: str,
    neutralized_body: str,
    transcription: str,
    client,
    model: str,
    temperature: float = 0.3
) -> str:
    """
    Generate a neutral factual summary.
    
    Args:
        title: Post title
        neutralized_body: Neutralized body text
        transcription: Formatted transcription
        client: OpenAI client
        model: Model name
        temperature: Sampling temperature
        
    Returns:
        Factual summary
    """
    prompt = SUMMARY_PROMPT.format(
        title=title,
        neutralized_body=neutralized_body,
        transcription=transcription
    )
    return call_llm(client, model, prompt, temperature=temperature, max_tokens=300)


# ---
# RECORD PROCESSING
# ---

def process_record(
    record: Dict[str, Any],
    client,
    model: str,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Process a single record: neutralize body and generate summary.
    
    Args:
        record: Input record dict
        client: OpenAI client
        model: Model name
        temperature: Sampling temperature
        
    Returns:
        Record with added neutralized_body and factual_summary fields
    """
    result = dict(record)
    
    # Get fields
    title = record.get("Title", "")
    body = record.get("Body", "")
    transcription_raw = record.get("Transcription", "")
    
    # Format transcription
    transcription_formatted = format_transcription(transcription_raw)
    result["transcription_formatted"] = transcription_formatted
    
    # Neutralize body
    try:
        neutralized_body = neutralize_body(body, client, model, temperature)
        result["neutralized_body"] = neutralized_body
    except Exception as e:
        print(f"    Error neutralizing body: {e}")
        result["neutralized_body"] = body  # Fallback to original
        result["neutralization_error"] = str(e)
    
    # Generate summary
    try:
        summary = generate_summary(
            title=title,
            neutralized_body=result["neutralized_body"],
            transcription=transcription_formatted,
            client=client,
            model=model,
            temperature=temperature
        )
        result["factual_summary"] = summary
    except Exception as e:
        print(f"    Error generating summary: {e}")
        result["factual_summary"] = f"{title}. {result['neutralized_body'][:200]}"  # Fallback
        result["summary_error"] = str(e)
    
    return result


# ---
# MAIN PIPELINE
# ---

def process_jsonl(
    input_file: str,
    output_file: str,
    model: str,
    api_key: str = None,
    temperature: float = 0.3,
    max_rows: int = None,
    skip_existing: bool = True
) -> Dict[str, int]:
    """
    Process JSONL file: neutralize bodies and generate summaries.
    
    Args:
        input_file: Path to input JSONL (from label extractor)
        output_file: Path to output JSONL
        model: LLM model name
        api_key: API key
        temperature: Sampling temperature
        max_rows: Maximum rows to process
        skip_existing: Skip records that already have summaries
        
    Returns:
        Statistics dict
    """
    load_dotenv()
    
    # Setup client
    client = create_client(
        api_key=api_key or os.getenv("OPENROUTER_API_KEY")
    )
    
    # Load input
    print(f"Loading {input_file}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if max_rows:
        records = records[:max_rows]
    
    print(f"Loaded {len(records)} records")
    print(f"  Model: {model}")
    print(f"  Temperature: {temperature}")
    
    # Check for existing output (for resume capability)
    existing_ids = set()
    if skip_existing and Path(output_file).exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        existing = json.loads(line)
                        if existing.get("factual_summary"):
                            existing_ids.add(existing.get("SubmissionID", existing.get("Index")))
                    except:
                        pass
        if existing_ids:
            print(f"  Skipping {len(existing_ids)} already processed records")
    
    # Filter records
    records_to_process = [
        r for r in records 
        if r.get("SubmissionID", r.get("Index")) not in existing_ids
    ]
    
    # Process
    stats = {
        "total": len(records),
        "processed": 0,
        "skipped": len(existing_ids),
        "errors": 0
    }
    
    # Open output file in append mode
    mode = 'a' if existing_ids else 'w'
    
    iterator = tqdm(records_to_process, desc="Processing") if HAS_TQDM else records_to_process
    
    for i, record in enumerate(iterator):
        try:
            result = process_record(record, client, model, temperature)
            
            # Write immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            stats["processed"] += 1
            
        except Exception as e:
            print(f"\n  Error on record {record.get('SubmissionID', i)}: {e}")
            stats["errors"] += 1
            
            # Write record with error flag
            record["processing_error"] = str(e)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        if not HAS_TQDM and (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(records_to_process)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARIZATION COMPLETE")
    print("=" * 60)
    print(f"  Total input:   {stats['total']}")
    print(f"  Processed:     {stats['processed']}")
    print(f"  Skipped:       {stats['skipped']}")
    print(f"  Errors:        {stats['errors']}")
    print("=" * 60)
    print(f"\nOutput saved to {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate neutralized bodies and factual summaries for AIO records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python aio_summarizer.py --input aio_labeled.jsonl --output aio_with_summaries.jsonl --model openai/gpt-4o-mini
  
  # Test on first 10 rows
  python aio_summarizer.py --input aio_labeled.jsonl --output aio_with_summaries.jsonl --model openai/gpt-4o-mini --max-rows 10
  
  # Resume interrupted processing
  python aio_summarizer.py --input aio_labeled.jsonl --output aio_with_summaries.jsonl --model openai/gpt-4o-mini --skip-existing
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file (from label extractor)")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--model", "-m", required=True, help="LLM model (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--api-key", help="API key (uses OPENROUTER_API_KEY env var if not set)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip records already in output file")
    
    args = parser.parse_args()
    
    process_jsonl(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_rows=args.max_rows,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
