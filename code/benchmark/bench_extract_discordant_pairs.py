#!/usr/bin/env python3
"""
Extract discordant pairs - records where C1 and C2 responses differ.

This script identifies cases where the model's answer changed between
C1 (factual) and C2 (dialogue) conditions.

Usage:
    python extract_discordant_pairs.py --input results/truthfulqa_sample_nova-lite-v1_Speaker1_vs_Speaker2_all.jsonl
    python extract_discordant_pairs.py --input results/ --all  # Process all JSONL files
"""

import os
import argparse
import glob
import pandas as pd
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_jsonl


def is_discordant(record: dict) -> tuple:
    """
    Check if a record has discordant C1 vs C2 responses.
    
    Returns:
        (is_discordant, comparison_type, c1_val, c2_val)
        comparison_type: 'true_correct' or 'false_incorrect'
    """
    # Normalize values (handle None, empty strings, etc.)
    def normalize(val):
        if val is None:
            return None
        return str(val).strip()
    
    c1_true = normalize(record.get('c1_true_statement_ans_t1'))
    c2_correct = normalize(record.get('c2_correct_chat_ans_t1'))
    c1_false = normalize(record.get('c1_false_statement_ans_t1'))
    c2_incorrect = normalize(record.get('c2_incorrect_chat_ans_t1'))
    
    discordant_pairs = []
    
    # Compare C1_True vs C2_Correct (both should be "1" when correct)
    if c1_true is not None and c2_correct is not None and c1_true != c2_correct:
        discordant_pairs.append(('true_correct', c1_true, c2_correct))
    
    # Compare C1_False vs C2_Incorrect (both should be "2" when correct)
    if c1_false is not None and c2_incorrect is not None and c1_false != c2_incorrect:
        discordant_pairs.append(('false_incorrect', c1_false, c2_incorrect))
    
    return discordant_pairs


def extract_discordant_records(filepath: str) -> pd.DataFrame:
    """
    Extract all discordant records from a JSONL file.
    
    Returns:
        DataFrame with discordant records plus comparison columns
    """
    records = load_jsonl(filepath)
    discordant_records = []
    
    for record in records:
        discordant_pairs = is_discordant(record)
        
        if discordant_pairs:
            # Create a row for each discordant pair
            for comp_type, c1_val, c2_val in discordant_pairs:
                row = record.copy()
                row['comparison_type'] = comp_type
                row['c1_answer'] = c1_val
                row['c2_answer'] = c2_val
                
                # Add which comparison this is
                if comp_type == 'true_correct':
                    row['c1_field'] = 'c1_true_statement_ans_t1'
                    row['c2_field'] = 'c2_correct_chat_ans_t1'
                    row['c1_expected'] = '1'
                    row['c2_expected'] = '1'
                else:  # false_incorrect
                    row['c1_field'] = 'c1_false_statement_ans_t1'
                    row['c2_field'] = 'c2_incorrect_chat_ans_t1'
                    row['c1_expected'] = '2'
                    row['c2_expected'] = '2'
                
                # Add correctness flags
                row['c1_correct'] = (c1_val == row['c1_expected'])
                row['c2_correct'] = (c2_val == row['c2_expected'])
                
                discordant_records.append(row)
    
    if not discordant_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(discordant_records)
    return df


def process_file(filepath: str, output_dir: str = None) -> pd.DataFrame:
    """Process a single file and return discordant records."""
    print(f"Processing {filepath}...")
    df = extract_discordant_records(filepath)
    
    if df.empty:
        print(f"  → No discordant pairs found")
        return df
    
    print(f"  → Found {len(df)} discordant records")
    
    # Save to CSV if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        input_name = Path(filepath).stem
        output_file = os.path.join(output_dir, f"{input_name}_discordant.csv")
        df.to_csv(output_file, index=False)
        print(f"  → Saved to {output_file}")
    
    return df


def process_directory(directory: str, pattern: str = "*_all.jsonl", output_dir: str = None) -> pd.DataFrame:
    """Process all matching files in a directory."""
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"No files matching '{pattern}' found in {directory}")
        return pd.DataFrame()
    
    print(f"Found {len(files)} files to process\n")
    
    all_discordant = []
    for filepath in sorted(files):
        df = process_file(filepath, output_dir)
        if not df.empty:
            df['source_file'] = Path(filepath).name
            all_discordant.append(df)
        print()
    
    if not all_discordant:
        print("No discordant pairs found in any file.")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_discordant, ignore_index=True)
    
    # Save combined results
    if output_dir:
        combined_file = os.path.join(output_dir, "all_discordant_pairs.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"Combined results saved to {combined_file}")
        print(f"  Total discordant records: {len(combined_df)}")
        
        # Print summary by dataset
        if 'dataset' in combined_df.columns:
            print("\nSummary by dataset:")
            summary = combined_df.groupby('dataset').size().sort_values(ascending=False)
            for dataset, count in summary.items():
                print(f"  {dataset}: {count} discordant records")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract discordant pairs (C1 vs C2 differences) from JSONL results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python extract_discordant_pairs.py --input results/truthfulqa_sample_nova-lite-v1_Speaker1_vs_Speaker2_all.jsonl

  # Process all files in directory
  python extract_discordant_pairs.py --input results/ --all

  # Specify output directory
  python extract_discordant_pairs.py --input results/ --all --output discordant_results/
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file or directory")
    parser.add_argument("--all", "-a", action="store_true", help="Process all JSONL files in directory")
    parser.add_argument("--pattern", "-p", default="*_all.jsonl", help="File pattern for directory mode (default: *_all.jsonl)")
    parser.add_argument("--output", "-o", help="Output directory for CSV files (default: same as input)")
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = args.output
    if output_dir is None:
        if os.path.isdir(args.input):
            output_dir = args.input
        else:
            output_dir = os.path.dirname(args.input) or "."
    
    if os.path.isdir(args.input) or args.all:
        directory = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
        process_directory(directory, args.pattern, output_dir)
    else:
        process_file(args.input, output_dir)


if __name__ == "__main__":
    main()

