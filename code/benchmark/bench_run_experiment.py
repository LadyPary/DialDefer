#!/usr/bin/env python3
"""
Run Turn 1 experiments on a JSONL dataset.

This script takes a JSONL file (formatted datasets) and runs the first turn
of the experiment with configurable model and speaker labels.

Usage:
    python run_turn1_jsonl.py --input data.jsonl --model openai/gpt-4o-mini
    python run_turn1_jsonl.py --input data.jsonl --model openai/gpt-4o-mini --speaker1 "User" --speaker2 "LLM"
    python run_turn1_jsonl.py --input data.jsonl --model openai/gpt-4o-mini --c2-only
"""

import os
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import create_client, chat, load_jsonl, append_jsonl, extract_json_from_response
from bench_prompts import c1_TF_factual_question_prompt, c2_convo_judge_prompt


def run_turn1_on_record(
    record: dict,
    client,
    model: str,
    speaker1_label: str,
    speaker2_label: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
    run_c1: bool = True,
    run_c2: bool = True,
    system_prompt: str = None
) -> dict:
    """
    Run Turn 1 experiment on a single JSONL record.
    
    Expected JSONL format:
    {
        "id": "...",
        "dataset": "...",
        "question": "...",
        "chosen_correct_answer": "...",
        "chosen_incorrect_answer": "...",
        ...
    }
    
    Returns dict with original fields + experiment results.
    """
    result = record.copy()
    
    # Extract fields from JSONL
    question = record.get("question", "")
    correct_answer = record.get("chosen_correct_answer", "")
    incorrect_answer = record.get("chosen_incorrect_answer", "")
    
    # Add speaker labels and system prompt to result
    result["speaker1"] = speaker1_label
    result["speaker2"] = speaker2_label
    result["system_prompt"] = system_prompt
    
    # C1_TRUE: Factual statement with correct answer
    if run_c1:
        c1_true_prompt = c1_TF_factual_question_prompt(question, correct_answer)
        c1_true_history = []
        
        try:
            c1_true_reply, c1_true_history = chat(
                client, model, c1_true_prompt,
                history=c1_true_history,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c1_true_response = extract_json_from_response(c1_true_reply)
        except Exception as e:
            print(f"[c1_true] Error on {record.get('id', '?')}: {e}")
            c1_true_response = {}
        
        result["c1_true_statement_ans_t1"] = c1_true_response.get("chosen_answer")
        result["c1_true_statement_reasoning_t1"] = c1_true_response.get("reasoning")
        result["c1_true_statement_t1_history"] = str(c1_true_history)
    
    # C1_FALSE: Factual statement with incorrect answer
    if run_c1:
        c1_false_prompt = c1_TF_factual_question_prompt(question, incorrect_answer)
        c1_false_history = []
        
        try:
            c1_false_reply, c1_false_history = chat(
                client, model, c1_false_prompt,
                history=c1_false_history,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c1_false_response = extract_json_from_response(c1_false_reply)
        except Exception as e:
            print(f"[c1_false] Error on {record.get('id', '?')}: {e}")
            c1_false_response = {}
        
        result["c1_false_statement_ans_t1"] = c1_false_response.get("chosen_answer")
        result["c1_false_statement_reasoning_t1"] = c1_false_response.get("reasoning")
        result["c1_false_statement_t1_history"] = str(c1_false_history)
    
    # C2_CORRECT: Correct dialogue
    if run_c2:
        c2_correct_prompt = c2_convo_judge_prompt(question, correct_answer, speaker1_label, speaker2_label)
        c2_correct_history = []
        
        try:
            c2_correct_reply, c2_correct_history = chat(
                client, model, c2_correct_prompt,
                history=c2_correct_history,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c2_correct_response = extract_json_from_response(c2_correct_reply)
        except Exception as e:
            print(f"[c2_correct] Error on {record.get('id', '?')}: {e}")
            c2_correct_response = {}
        
        result["c2_correct_chat_ans_t1"] = c2_correct_response.get("chosen_answer")
        result["c2_correct_chat_reasoning_t1"] = c2_correct_response.get("reasoning")
        result["c2_correct_chat_t1_history"] = str(c2_correct_history)
    
    # C2_INCORRECT: Incorrect dialogue
    if run_c2:
        c2_incorrect_prompt = c2_convo_judge_prompt(question, incorrect_answer, speaker1_label, speaker2_label)
        c2_incorrect_history = []
        
        try:
            c2_incorrect_reply, c2_incorrect_history = chat(
                client, model, c2_incorrect_prompt,
                history=c2_incorrect_history,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_prompt
            )
            c2_incorrect_response = extract_json_from_response(c2_incorrect_reply)
        except Exception as e:
            print(f"[c2_incorrect] Error on {record.get('id', '?')}: {e}")
            c2_incorrect_response = {}
        
        result["c2_incorrect_chat_ans_t1"] = c2_incorrect_response.get("chosen_answer")
        result["c2_incorrect_chat_reasoning_t1"] = c2_incorrect_response.get("reasoning")
        result["c2_incorrect_chat_t1_history"] = str(c2_incorrect_history)
    
    return result


def run_experiment(
    input_file: str,
    output_file: str,
    model: str,
    speaker1_label: str = "Speaker 1",
    speaker2_label: str = "Speaker 2",
    temperature: float = 0.0,
    max_tokens: int = 300,
    run_c1: bool = True,
    run_c2: bool = True,
    max_rows: Optional[int] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> list[dict]:
    """
    Run Turn 1 experiment on entire JSONL dataset.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        model: Model name (e.g., "openai/gpt-4o-mini")
        speaker1_label: Label for questioning speaker
        speaker2_label: Label for answering speaker
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        run_c1: Whether to run C1 conditions
        run_c2: Whether to run C2 conditions
        max_rows: Maximum rows to process (None = all)
        api_key: API key (uses env var if not provided)
    
    Returns:
        DataFrame with all results
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key (check both OPENAI_API_KEY and OPENROUTER_API_KEY)
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY, or pass --api-key")
    
    # Create client
    client = create_client(api_key)
    
    # Load JSONL
    print(f"Loading {input_file}...")
    records = load_jsonl(input_file)
    print(f"Loaded {len(records)} records")
    
    # Limit rows if specified
    if max_rows is not None:
        records = records[:max_rows]
        print(f"  → Processing first {max_rows} rows only")
    
    # Extract dataset name from first record or filename
    dataset_name = records[0].get("dataset", Path(input_file).stem) if records else "unknown"
    
    # Print configuration
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"  Input:    {input_file}")
    print(f"  Output:   {output_file}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  Model:    {model}")
    print(f"  Speaker1: {speaker1_label}")
    print(f"  Speaker2: {speaker2_label}")
    print(f"  Run C1:   {run_c1}")
    print(f"  Run C2:   {run_c2}")
    print(f"  Rows:     {len(records)}")
    if system_prompt:
        print(f"  System:   {system_prompt[:50]}..." if len(system_prompt) > 50 else f"  System:   {system_prompt}")
    print("="*70 + "\n")
    
    # Process records
    results = []
    iterator = tqdm(records, desc="Processing") if HAS_TQDM else records
    
    for i, record in enumerate(iterator):
        try:
            result = run_turn1_on_record(
                record=record,
                client=client,
                model=model,
                speaker1_label=speaker1_label,
                speaker2_label=speaker2_label,
                temperature=temperature,
                max_tokens=max_tokens,
                run_c1=run_c1,
                run_c2=run_c2,
                system_prompt=system_prompt
            )
            
            # Write row immediately (crash-safe)
            append_jsonl(result, output_file)
            results.append(result)
            
        except Exception as e:
            print(f"ERROR processing record {i} ({record.get('id', '?')}): {e}")
            continue
        
        if not HAS_TQDM and (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(records)} rows")
    
    print(f"\nExperiment complete. Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Turn 1 experiments on a JSONL dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_turn1_jsonl.py --input data/truthfulqa.jsonl --model openai/gpt-4o-mini

  # Run C2-only with custom speaker labels
  python run_turn1_jsonl.py --input data/truthfulqa.jsonl --model openai/gpt-4o-mini \\
      --speaker1 "User" --speaker2 "LLM" --c2-only

  # Test on first 10 rows
  python run_turn1_jsonl.py --input data/truthfulqa.jsonl --model openai/gpt-4o-mini --max-rows 10

  # Run C1-only (baseline factual)
  python run_turn1_jsonl.py --input data/truthfulqa.jsonl --model openai/gpt-4o-mini --c1-only
        """
    )
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g., openai/gpt-4o-mini)")
    
    # Optional arguments
    parser.add_argument("--output", "-o", help="Output CSV file path (auto-generated if not specified)")
    parser.add_argument("--speaker1", default="Speaker 1", help="Label for questioning speaker (default: Speaker 1)")
    parser.add_argument("--speaker2", default="Speaker 2", help="Label for answering speaker (default: Speaker 2)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens per response (default: 300)")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to process (for testing)")
    parser.add_argument("--api-key", help="OpenRouter API key (uses OPENROUTER_API_KEY env var if not specified)")
    parser.add_argument("--system-prompt", help="System prompt to add to all API calls")
    
    # Condition selection
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument("--c1-only", action="store_true", help="Run only C1 conditions (factual)")
    condition_group.add_argument("--c2-only", action="store_true", help="Run only C2 conditions (dialogue)")
    
    args = parser.parse_args()
    
    # Determine which conditions to run
    run_c1 = not args.c2_only
    run_c2 = not args.c1_only
    
    # Auto-generate output filename if not specified
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        dataset_name = input_path.stem.replace("_formatted", "").replace("_sampled", "")
        model_short = args.model.split("/")[-1]
        speaker_tag = f"{args.speaker1}_vs_{args.speaker2}".replace(" ", "")
        
        conditions = []
        if run_c1 and run_c2:
            conditions.append("all")
        elif run_c1:
            conditions.append("c1only")
        else:
            conditions.append("c2only")
        
        output_file = f"results/{dataset_name}_{model_short}_{speaker_tag}_{conditions[0]}.jsonl"
    
    # Run experiment
    run_experiment(
        input_file=args.input,
        output_file=output_file,
        model=args.model,
        speaker1_label=args.speaker1,
        speaker2_label=args.speaker2,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        run_c1=run_c1,
        run_c2=run_c2,
        max_rows=args.max_rows,
        api_key=args.api_key,
        system_prompt=args.system_prompt
    )


if __name__ == "__main__":
    main()
