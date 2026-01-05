#!/usr/bin/env python3
"""
Format processed AIO data for the C1/C2 experiment.

MAIN EXPERIMENT: Dialogic Format Effect
- C1: Context + Narrative (collapsed description)
- C2: Context + Dialogue (turn-by-turn)
- Speaker B = OP (the one who may be overreacting)

Input: Output from aio_transcript_processor.py
Output: JSONL ready for aio_run_experiment_v2.py

Usage:
    python aio_formatter_v2.py --input aio_processed.jsonl --output aio_experiment_ready.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def format_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a processed record for the experiment.
    
    Required fields from processor:
    - transcript_prose: C1 input (narrative format)
    - transcript_dialogue: C2 input (dialogue format)
    - ground_truth: OR or NOR
    - neutralized_body: Context
    - Title: Context
    """
    # Get ground truth
    gt_label = record.get("ground_truth", "")
    if gt_label not in ("OR", "NOR"):
        # Try to infer from other fields
        label_extracted = record.get("label_extracted", "")
        if "OR" in str(label_extracted).upper() and "NOR" not in str(label_extracted).upper():
            gt_label = "OR"
        elif "NOR" in str(label_extracted).upper() or "NOT" in str(label_extracted).upper():
            gt_label = "NOR"
        else:
            gt_label = None
    
    # Determine correct/incorrect answers based on ground truth
    # Speaker B = OP, so if GT=OR, Speaker B IS overreacting
    if gt_label == "OR":
        correct_answer = "overreacting"
        incorrect_answer = "not overreacting"
    elif gt_label == "NOR":
        correct_answer = "not overreacting"
        incorrect_answer = "overreacting"
    else:
        correct_answer = None
        incorrect_answer = None
    
    # Build context from neutralized title + body
    # Use neutralized_context if available (third-person), else fall back to original
    context = record.get("neutralized_context", "")
    
    if not context:
        # Fallback to original (not ideal - will have first-person)
        title = record.get("Title", "")
        body = record.get("Body", "")
        context_parts = []
        if title:
            context_parts.append(title)
        if body:
            context_parts.append(body)
        context = "\n\n".join(context_parts)
    
    # Build formatted record
    formatted = {
        "id": record.get("SubmissionID", record.get("id", "")),
        "dataset": "aio",
        
        # Label type (for downstream use)
        "label_type": record.get("label_type", "speaker"),
        
        # Context (same for C1 and C2)
        "context": context,
        
        # C1 input: narrative description of conversation
        "c1_input": record.get("transcript_prose", ""),
        
        # C2 input: dialogue format of conversation
        "c2_input": record.get("transcript_dialogue", ""),
        
        # Ground truth and answers
        "ground_truth": gt_label,
        "chosen_correct_answer": correct_answer,
        "chosen_incorrect_answer": incorrect_answer,
        
        # Metadata
        "title": record.get("Title", ""),
        "speaker_a_identity": record.get("speaker_a_identity", ""),
        
        # Original data for reference
        "original_transcript": record.get("transcription_formatted", record.get("Transcription", "")),
        "original_body": record.get("Body", ""),
    }
    
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Format processed AIO data for experiment"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL from processor")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL for experiment")
    parser.add_argument("--require-both", action="store_true", 
                        help="Only include records with both narrative and dialogue")
    
    args = parser.parse_args()
    
    # Load input
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records")
    
    # Format records
    formatted = []
    skipped = {"no_narrative": 0, "no_dialogue": 0, "no_gt": 0, "no_context": 0}
    
    for record in records:
        fmt = format_record(record)
        
        # Validation
        if not fmt["c1_input"]:
            skipped["no_narrative"] += 1
            if args.require_both:
                continue
        
        if not fmt["c2_input"]:
            skipped["no_dialogue"] += 1
            if args.require_both:
                continue
        
        if not fmt["context"]:
            skipped["no_context"] += 1
            continue
        
        if not fmt["ground_truth"]:
            skipped["no_gt"] += 1
            continue  # Always skip records without ground truth
        
        formatted.append(fmt)
    
    # Save output
    with open(args.output, 'w') as f:
        for record in formatted:
            f.write(json.dumps(record) + '\n')
    
    print(f"\nSaved {len(formatted)} records to {args.output}")
    print(f"  Skipped - no narrative: {skipped['no_narrative']}")
    print(f"  Skipped - no dialogue: {skipped['no_dialogue']}")
    print(f"  Skipped - no context: {skipped['no_context']}")
    print(f"  Skipped - no ground truth: {skipped['no_gt']}")
    
    # Preview
    if formatted:
        print("\n" + "=" * 60)
        print("PREVIEW: First record")
        print("=" * 60)
        r = formatted[0]
        print(f"ID: {r['id']}")
        print(f"Ground Truth: {r['ground_truth']}")
        print(f"Correct answer: Speaker B is {r['chosen_correct_answer']}")
        print(f"\nContext (first 200 chars):\n{r['context'][:200]}...")
        print(f"\nC1 Input - Narrative (first 300 chars):\n{r['c1_input'][:300]}...")
        print(f"\nC2 Input - Dialogue (first 300 chars):\n{r['c2_input'][:300]}...")


if __name__ == "__main__":
    main()
