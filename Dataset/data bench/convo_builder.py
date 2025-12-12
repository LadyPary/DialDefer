#!/usr/bin/env python3
"""
Conversation builder for sampled JSONL files.
Adds columns: Question, Chosen Correct Answer, Chosen Incorrect Answer, correct_chat, incorrect_chat
"""

import json
from pathlib import Path


def build_chat(question: str, answer: str) -> str:
    """Build a chat conversation between Speaker 1 and Speaker 2."""
    return f"Speaker 1: {question}\nSpeaker 2: {answer}"


def get_incorrect_answer(record: dict) -> str:
    """
    Get the chosen incorrect answer.
    For advisorqa: pick the one with the lowest reward (most wrong).
    For others: pick the first one.
    """
    if record['dataset'] == 'advisorqa':
        rewards = record['meta'].get('reward', [])
        incorrect_answers = record['incorrect_answers']
        # Rewards are ordered: first is correct answer, rest are incorrect
        # So incorrect rewards start at index 1
        incorrect_rewards = rewards[1:] if len(rewards) > 1 else []
        if incorrect_rewards and len(incorrect_rewards) == len(incorrect_answers):
            # Find index of minimum reward
            min_idx = incorrect_rewards.index(min(incorrect_rewards))
            return incorrect_answers[min_idx]
    return record['incorrect_answers'][0]


def process_record(record: dict) -> dict:
    """Process a single record and add the new columns."""
    question = record['question']
    chosen_correct = record['correct_answers'][0]
    chosen_incorrect = get_incorrect_answer(record)
    
    return {
        **record,
        'question': question,
        'chosen_correct_answer': chosen_correct,
        'chosen_incorrect_answer': chosen_incorrect,
        'correct_chat': build_chat(question, chosen_correct),
        'incorrect_chat': build_chat(question, chosen_incorrect),
    }


def process_jsonl(input_path: Path, output_path: Path) -> None:
    """Process a JSONL file and add conversation columns."""
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(process_record(record))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Processed {len(records)} records: {output_path.name}")


def main():
    sampled_dir = Path(__file__).parent / 'full datasets'
    output_dir = Path(__file__).parent / 'formatted'
    output_dir.mkdir(exist_ok=True)
    
    for input_file in sampled_dir.glob('*_unified.jsonl'):
        output_file = output_dir / input_file.name.replace('_unified', '_formatted')
        process_jsonl(input_file, output_file)
    
    print(f"\n🎉 Done! All formatted files saved to {output_dir}")


if __name__ == '__main__':
    main()
