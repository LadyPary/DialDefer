#!/usr/bin/env python3
"""
Deterministic sampler for JSONL datasets.
Supports both initial sampling and extended sampling with exclusion of existing IDs.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

SEED = 42
SAMPLES_PER_FILE = 100
DATA_DIR = Path(__file__).parent

# Target totals for extended sampling (including initial 100)
TARGET_SAMPLES = {
    'advisorqa': 300,
    'amqa': 240,
    'bbq': 300,
    'gpqa': 134,
    'halueval_qa': 300,
    'harp_mcq': 300,
    'plausibleqa': 300,
    'socialiqa': 300,
}


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


def get_existing_ids(filepath: Path) -> set:
    """Get IDs from already sampled file."""
    if not filepath.exists():
        return set()
    records = load_jsonl(filepath)
    return {record['id'] for record in records}


def sample_random(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Random sampling without balancing."""
    if exclude_ids:
        records = [r for r in records if r['id'] not in exclude_ids]
    return rng.sample(records, min(n, len(records)))


def sample_balanced(records: list[dict], key_func, n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Balanced sampling across categories defined by key_func."""
    # Filter out excluded IDs
    if exclude_ids:
        records = [r for r in records if r['id'] not in exclude_ids]
    
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


def sample_advisorqa(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Balanced by meta.label"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['label'],
        n=n,
        rng=rng,
        exclude_ids=exclude_ids
    )


def sample_amqa(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Random sampling"""
    return sample_random(records, n, rng, exclude_ids)


def sample_bbq(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Balanced by meta.context_condition"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['context_condition'],
        n=n,
        rng=rng,
        exclude_ids=exclude_ids
    )


def sample_gpqa(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Balanced by meta.domain"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['domain'],
        n=n,
        rng=rng,
        exclude_ids=exclude_ids
    )


def sample_halueval_qa(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Random sampling"""
    return sample_random(records, n, rng, exclude_ids)


def sample_harp_mcq(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Balanced by meta.level"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['level'],
        n=n,
        rng=rng,
        exclude_ids=exclude_ids
    )


def sample_plausibleqa(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Random sampling"""
    return sample_random(records, n, rng, exclude_ids)


def sample_socialiqa(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Balanced by meta.promptDim"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['promptDim'],
        n=n,
        rng=rng,
        exclude_ids=exclude_ids
    )


def sample_truthfulqa(records: list[dict], n: int, rng: random.Random, exclude_ids: set = None) -> list[dict]:
    """Balanced by meta.type"""
    return sample_balanced(
        records,
        key_func=lambda r: r['meta']['type'],
        n=n,
        rng=rng,
        exclude_ids=exclude_ids
    )


# Mapping of dataset names to their sampling functions
SAMPLERS = {
    'advisorqa': sample_advisorqa,
    'amqa': sample_amqa,
    'bbq': sample_bbq,
    'gpqa': sample_gpqa,
    'halueval_qa': sample_halueval_qa,
    'harp_mcq': sample_harp_mcq,
    'plausibleqa': sample_plausibleqa,
    'socialiqa': sample_socialiqa,
    'truthfulqa': sample_truthfulqa,
}


def main():
    parser = argparse.ArgumentParser(description='Sample datasets with optional exclusion of existing IDs')
    parser.add_argument('--extend', action='store_true', 
                       help='Extended mode: sample additional records excluding prev_sampled/')
    parser.add_argument('--initial', action='store_true',
                       help='Initial mode: sample first 100 from unified files (default)')
    args = parser.parse_args()
    
    # Default to initial mode if neither specified
    if not args.extend and not args.initial:
        args.initial = True
    
    if args.extend:
        # EXTENDED MODE: Sample more, excluding prev_sampled
        formatted_dir = DATA_DIR / 'datasets' / 'formatted'
        prev_sampled_dir = DATA_DIR / 'prev_sampled'
        output_dir = DATA_DIR / 'sampled'
        output_dir.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("EXTENDED SAMPLING MODE")
        print("Excluding previously sampled IDs from prev_sampled/")
        print("=" * 70 + "\n")
        
        for dataset_name, sampler_func in SAMPLERS.items():
            if dataset_name == 'truthfulqa':
                continue  # Skip truthfulqa in extended mode
            
            formatted_file = formatted_dir / f"{dataset_name}_formatted.jsonl"
            prev_sampled_file = prev_sampled_dir / f"{dataset_name}_sampled.jsonl"
            output_file = output_dir / f"{dataset_name}_sampled.jsonl"
            
            if not formatted_file.exists():
                print(f"⚠️  Skipping {dataset_name} (formatted file not found)")
                continue
            
            # Get existing IDs
            existing_ids = get_existing_ids(prev_sampled_file)
            initial_count = len(existing_ids)
            
            # Calculate additional needed
            target_total = TARGET_SAMPLES[dataset_name]
            additional_needed = target_total - initial_count
            
            if additional_needed <= 0:
                print(f"📂 {dataset_name}")
                print(f"   Already have {initial_count} samples (target: {target_total})")
                print(f"   ⏭️  Skipping\n")
                continue
            
            rng = random.Random(SEED)
            
            print(f"📂 {dataset_name}")
            print(f"   Previously sampled: {initial_count}")
            print(f"   Target total: {target_total}")
            print(f"   Additional needed: {additional_needed}")
            
            all_records = load_jsonl(formatted_file)
            print(f"   Available in formatted: {len(all_records)}")
            
            # Sample additional
            new_samples = sampler_func(all_records, additional_needed, rng, existing_ids)
            
            # Combine with existing
            existing_samples = load_jsonl(prev_sampled_file) if prev_sampled_file.exists() else []
            combined_samples = existing_samples + new_samples
            
            save_jsonl(combined_samples, output_file)
            print(f"   ✅ Saved {len(combined_samples)} total ({len(new_samples)} new)")
            print(f"      → {output_file}\n")
        
        print("=" * 70)
        print(f"🎉 Extended sampling complete! Results in {output_dir}")
        print("=" * 70)
    
    else:
        # INITIAL MODE: Original behavior for unified files
        output_dir = DATA_DIR / 'sampled'
        output_dir.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("INITIAL SAMPLING MODE (100 samples per dataset)")
        print("=" * 70 + "\n")
        
        for dataset_name, sampler_func in SAMPLERS.items():
            filename = f"{dataset_name}_unified.jsonl"
            filepath = DATA_DIR / filename
            if not filepath.exists():
                print(f"⚠️  Skipping {filename} (file not found)")
                continue
            
            rng = random.Random(SEED)
            
            print(f"📂 Processing {filename}...")
            records = load_jsonl(filepath)
            sampled = sampler_func(records, SAMPLES_PER_FILE, rng)
            
            output_path = output_dir / f"{dataset_name}_sampled.jsonl"
            save_jsonl(sampled, output_path)
            print(f"   ✅ Saved {len(sampled)} samples to {output_path.name}\n")
        
        print(f"🎉 Done! All samples saved to {output_dir}")


if __name__ == '__main__':
    main()
