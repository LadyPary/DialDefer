#!/usr/bin/env python3
"""
Run AIO experiment: Dialogue Format Effect

MAIN EXPERIMENT:
- C1: Conversation as PROSE narrative
- C2: Conversation as DIALOGUE format
- Same content, only format differs

This directly tests if dialogue format shifts model judgment.

Usage:
    python aio_run_experiment.py --input aio_experiment_ready.jsonl --model openai/gpt-4o-mini
"""

import os
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import create_client, chat
from aio_prompts import c1_aio_factual_prompt, c2_aio_conversation_prompt
from aio_labels import get_labels, list_label_types


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not response:
        return {}
    
    # Try to find JSON in code blocks first
    if "```" in response:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
    
    # Clean up common issues
    response = response.strip()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}


# ---
# EXPERIMENT RUNNER
# ---

def run_experiment_on_record(
    record: dict,
    client,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
    labels: dict = None,
    save_history: bool = True,
    system_prompt: str = None
) -> dict:
    """
    Run C1/C2 experiment on a single record.
    
    C1: Context + Narrative → "Is the statement correct? {label_b} is {X}"
    C2: Context + Dialogue → "Is the statement correct? {label_b} is {X}"
    
    Same content, different format. Tests dialogic format effect.
    """
    result = record.copy()
    
    # Get labels from record or use provided/default
    if labels is None:
        label_type = record.get("label_type", "speaker")
        labels = get_labels(label_type)
    
    label_b = labels["B"]
    
    # Extract inputs
    context = record.get("context", "")
    c1_input = record.get("c1_input", "")  # Narrative
    c2_input = record.get("c2_input", "")  # Dialogue
    correct_judgment = record.get("chosen_correct_answer", "")
    incorrect_judgment = record.get("chosen_incorrect_answer", "")
    
    # ========== C1_TRUE: Narrative + CORRECT judgment ==========
    if c1_input:
        c1_true_prompt = c1_aio_factual_prompt(context, c1_input, correct_judgment, label_b)
        try:
            c1_true_reply, _ = chat(
                client, model, c1_true_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c1_true_response = extract_json_from_response(c1_true_reply)
        except Exception as e:
            print(f"[c1_true] Error on {record.get('id', '?')}: {e}")
            c1_true_reply = ""
            c1_true_response = {}
        
        result["c1_true_statement_ans_t1"] = c1_true_response.get("chosen_answer")
        result["c1_true_statement_reasoning_t1"] = c1_true_response.get("reasoning")
        if save_history:
            result["c1_true_statement_history_t1"] = [
                {"role": "user", "content": c1_true_prompt},
                {"role": "assistant", "content": c1_true_reply}
            ]
    
    # ========== C1_FALSE: Narrative + INCORRECT judgment ==========
    if c1_input:
        c1_false_prompt = c1_aio_factual_prompt(context, c1_input, incorrect_judgment, label_b)
        try:
            c1_false_reply, _ = chat(
                client, model, c1_false_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c1_false_response = extract_json_from_response(c1_false_reply)
        except Exception as e:
            print(f"[c1_false] Error on {record.get('id', '?')}: {e}")
            c1_false_reply = ""
            c1_false_response = {}
        
        result["c1_false_statement_ans_t1"] = c1_false_response.get("chosen_answer")
        result["c1_false_statement_reasoning_t1"] = c1_false_response.get("reasoning")
        if save_history:
            result["c1_false_statement_history_t1"] = [
                {"role": "user", "content": c1_false_prompt},
                {"role": "assistant", "content": c1_false_reply}
            ]
    
    # ========== C2_CORRECT: Dialogue + CORRECT judgment ==========
    if c2_input:
        c2_correct_prompt = c2_aio_conversation_prompt(context, c2_input, correct_judgment, label_b)
        try:
            c2_correct_reply, _ = chat(
                client, model, c2_correct_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c2_correct_response = extract_json_from_response(c2_correct_reply)
        except Exception as e:
            print(f"[c2_correct] Error on {record.get('id', '?')}: {e}")
            c2_correct_reply = ""
            c2_correct_response = {}
        
        result["c2_correct_chat_ans_t1"] = c2_correct_response.get("chosen_answer")
        result["c2_correct_chat_reasoning_t1"] = c2_correct_response.get("reasoning")
        if save_history:
            result["c2_correct_chat_history_t1"] = [
                {"role": "user", "content": c2_correct_prompt},
                {"role": "assistant", "content": c2_correct_reply}
            ]
    
    # ========== C2_INCORRECT: Dialogue + INCORRECT judgment ==========
    if c2_input:
        c2_incorrect_prompt = c2_aio_conversation_prompt(context, c2_input, incorrect_judgment, label_b)
        try:
            c2_incorrect_reply, _ = chat(
                client, model, c2_incorrect_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c2_incorrect_response = extract_json_from_response(c2_incorrect_reply)
        except Exception as e:
            print(f"[c2_incorrect] Error on {record.get('id', '?')}: {e}")
            c2_incorrect_reply = ""
            c2_incorrect_response = {}
        
        result["c2_incorrect_chat_ans_t1"] = c2_incorrect_response.get("chosen_answer")
        result["c2_incorrect_chat_reasoning_t1"] = c2_incorrect_response.get("reasoning")
        if save_history:
            result["c2_incorrect_chat_history_t1"] = [
                {"role": "user", "content": c2_incorrect_prompt},
                {"role": "assistant", "content": c2_incorrect_reply}
            ]
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run AIO Dialogue Format Effect experiment"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", help="Output JSONL file (default: auto-generated)")
    parser.add_argument("--model", "-m", default="openai/gpt-4o-mini", help="Model to use")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to process")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens per response (default: 300)")
    parser.add_argument("--system-prompt", help="System prompt to add to all API calls")
    parser.add_argument("--label-type", "-l", default=None,
                        choices=list_label_types(),
                        help=f"Override label type. Choices: {list_label_types()}. "
                             "If not set, uses label_type from input data.")
    parser.add_argument("--speaker-a", help="Custom label for Speaker A (overrides label-type)")
    parser.add_argument("--speaker-b", help="Custom label for Speaker B (overrides label-type)")
    parser.add_argument("--no-history", action="store_true",
                        help="Don't save conversation history (smaller output files)")
    
    args = parser.parse_args()
    
    save_history = not args.no_history
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")
    
    client = create_client(api_key)
    
    # Get labels if overriding
    if args.speaker_a or args.speaker_b:
        # Custom speaker labels override everything
        labels = {
            "A": args.speaker_a or "Speaker A",
            "B": args.speaker_b or "Speaker B"
        }
    elif args.label_type:
        labels = get_labels(args.label_type)
    else:
        labels = None
    
    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        model_name = args.model.split("/")[-1]
        input_stem = Path(args.input).stem
        label_suffix = f"_{args.label_type}" if args.label_type else ""
        output_path = f"{input_stem}_{model_name}{label_suffix}_results.jsonl"
    
    # Load input
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if args.max_rows:
        records = records[:args.max_rows]
    
    # Determine label type for display
    if args.speaker_a or args.speaker_b:
        display_labels = {
            "A": args.speaker_a or "Speaker A",
            "B": args.speaker_b or "Speaker B"
        }
        display_label_type = "custom"
    elif args.label_type:
        display_label_type = args.label_type
        display_labels = get_labels(display_label_type)
    else:
        display_label_type = records[0].get("label_type", "speaker") if records else "speaker"
        display_labels = get_labels(display_label_type)
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing results (resume capability)
    existing_ids = set()
    if Path(output_path).exists():
        print(f"Found existing output file: {output_path}")
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    existing_ids.add(result.get('id'))
        if existing_ids:
            print(f"  → {len(existing_ids)} records already processed, will skip them")
    
    # Filter out already processed records
    records_to_process = [r for r in records if r.get('id') not in existing_ids]
    
    print(f"Running experiment on {len(records_to_process)} records (total: {len(records)})")
    print(f"Model: {args.model}")
    print(f"Labels: {display_labels['A']} / {display_labels['B']} (type: {display_label_type})")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt[:50]}..." if len(args.system_prompt) > 50 else f"System prompt: {args.system_prompt}")
    print(f"Save history: {save_history}")
    print(f"Output: {output_path}")
    print()
    
    if not records_to_process:
        print("All records already processed!")
        return
    
    # Run experiment with incremental saving
    results = []
    iterator = tqdm(records_to_process, desc="Running") if HAS_TQDM else records_to_process
    
    # Open output file for incremental writing (append mode)
    with open(output_path, 'a') as f:
        for i, record in enumerate(iterator):
            if not HAS_TQDM:
                print(f"Processing {i+1}/{len(records)}: {record.get('id', '?')}")
            
            result = run_experiment_on_record(
                record, client, args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                labels=labels,
                save_history=save_history,
                system_prompt=args.system_prompt
            )
            results.append(result)
            
            # Write result immediately (incremental save)
            f.write(json.dumps(result) + '\n')
            f.flush()  # Ensure it's written to disk
    
    print(f"\nSaved results to {output_path}")
    
    # Load ALL results for summary (including previously processed ones)
    all_results = []
    with open(output_path, 'r') as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))
    
    # Quick summary
    print("\n" + "=" * 60)
    print(f"QUICK RESULTS ({len(all_results)} total records)")
    print("=" * 60)
    
    c1_true_acc = sum(1 for r in all_results if r.get('c1_true_statement_ans_t1') == '1') / len(all_results) * 100
    c1_false_acc = sum(1 for r in all_results if r.get('c1_false_statement_ans_t1') == '2') / len(all_results) * 100
    c2_correct_acc = sum(1 for r in all_results if r.get('c2_correct_chat_ans_t1') == '1') / len(all_results) * 100
    c2_incorrect_acc = sum(1 for r in all_results if r.get('c2_incorrect_chat_ans_t1') == '2') / len(all_results) * 100
    
    print(f"C1 (Prose):    True={c1_true_acc:.1f}%, False={c1_false_acc:.1f}%, Avg={(c1_true_acc+c1_false_acc)/2:.1f}%")
    print(f"C2 (Dialogue): Corr={c2_correct_acc:.1f}%, Incorr={c2_incorrect_acc:.1f}%, Avg={(c2_correct_acc+c2_incorrect_acc)/2:.1f}%")
    
    delta_correct = c2_correct_acc - c1_true_acc
    delta_incorrect = c2_incorrect_acc - c1_false_acc
    dds = delta_correct - delta_incorrect
    
    print(f"\nΔ_Correct: {delta_correct:+.1f}%")
    print(f"Δ_Incorrect: {delta_incorrect:+.1f}%")
    print(f"DDS: {dds:+.1f}")


if __name__ == "__main__":
    main()
