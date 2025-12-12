#!/usr/bin/env python3
"""
Deterministic sampler for JSONL datasets.
Samples 100 records from each dataset with specific balancing criteria.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
SAMPLES_PER_FILE = 100
DATA_DIR = Path(__file__).parent


def load_jsonl(filepath: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def save_jsonl(records: list[dict], filepath: Path) -> None:
    """Save records to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def sample_random(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Random sampling without balancing."""
    return rng.sample(records, min(n, len(records)))


def sample_balanced(records: list[dict], key_func, n: int, rng: random.Random) -> list[dict]:
    """Balanced sampling across categories defined by key_func."""
    # Group records by category
    groups = defaultdict(list)
    for record in records:
        category = key_func(record)
        groups[category].append(record)
    
    categories = list(groups.keys())
    num_categories = len(categories)
    per_category = n // num_categories
    remainder = n % num_categories
    
    sampled = []
    # Shuffle categories to randomly distribute remainder
    rng.shuffle(categories)
    
    for i, category in enumerate(categories):
        # Distribute remainder across first few categories
        take = per_category + (1 if i < remainder else 0)
        available = groups[category]
        take = min(take, len(available))
        sampled.extend(rng.sample(available, take))
    
    return sampled


def sample_advisorqa(records: list[dict], rng: random.Random) -> list[dict]:
    """50 safe + 50 unsafe balanced by meta.label"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['label'],
        n=SAMPLES_PER_FILE,
        rng=rng
    )


def sample_amqa(records: list[dict], rng: random.Random) -> list[dict]:
    """Random sampling"""
    return sample_random(records, SAMPLES_PER_FILE, rng)


def sample_bbq(records: list[dict], rng: random.Random) -> list[dict]:
    """50 ambig + 50 disambig balanced by meta.context_condition"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['context_condition'],
        n=SAMPLES_PER_FILE,
        rng=rng
    )


def sample_gpqa(records: list[dict], rng: random.Random) -> list[dict]:
    """~33 each domain balanced by meta.domain"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['domain'],
        n=SAMPLES_PER_FILE,
        rng=rng
    )


def sample_halueval_qa(records: list[dict], rng: random.Random) -> list[dict]:
    """Random sampling"""
    return sample_random(records, SAMPLES_PER_FILE, rng)


def sample_harp_mcq(records: list[dict], rng: random.Random) -> list[dict]:
    """25 per level (1,2,3,4) balanced by meta.level"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['level'],
        n=SAMPLES_PER_FILE,
        rng=rng
    )


def sample_plausibleqa(records: list[dict], rng: random.Random) -> list[dict]:
    """Random sampling"""
    return sample_random(records, SAMPLES_PER_FILE, rng)


def sample_socialiqa(records: list[dict], rng: random.Random) -> list[dict]:
    """~11 per promptDim balanced by meta.promptDim"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['promptDim'],
        n=SAMPLES_PER_FILE,
        rng=rng
    )


def sample_truthfulqa(records: list[dict], rng: random.Random) -> list[dict]:
    """50 Adversarial + 50 Non-Adversarial balanced by meta.type"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['type'],
        n=SAMPLES_PER_FILE,
        rng=rng
    )


# Mapping of input files to their sampling functions
SAMPLERS = {
    'advisorqa_unified.jsonl': sample_advisorqa,
    'amqa_unified.jsonl': sample_amqa,
    'bbq_unified.jsonl': sample_bbq,
    'gpqa_unified.jsonl': sample_gpqa,
    'halueval_qa_unified.jsonl': sample_halueval_qa,
    'harp_mcq_unified.jsonl': sample_harp_mcq,
    'plausibleqa_unified.jsonl': sample_plausibleqa,
    'socialiqa_unified.jsonl': sample_socialiqa,
    'truthfulqa_unified.jsonl': sample_truthfulqa,
}


def main():
    output_dir = DATA_DIR / 'sampled'
    output_dir.mkdir(exist_ok=True)
    
    for filename, sampler_func in SAMPLERS.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"⚠️  Skipping {filename} (file not found)")
            continue
        
        # Use fresh RNG with same seed for each file for reproducibility
        rng = random.Random(SEED)
        
        print(f"📂 Processing {filename}...")
        records = load_jsonl(filepath)
        sampled = sampler_func(records, rng)
        
        output_path = output_dir / filename.replace('_unified', '_sampled')
        save_jsonl(sampled, output_path)
        print(f"   ✅ Saved {len(sampled)} samples to {output_path.name}")
    
    print(f"\n🎉 Done! All samples saved to {output_dir}")


if __name__ == '__main__':
    main()
