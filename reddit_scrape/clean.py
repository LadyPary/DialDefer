import argparse
import json
import pandas as pd

DEFAULT_INPUT_PATH = "AIODataFull.csv"
DEFAULT_OUTPUT_PATH = "AIODataFullClean.csv"

TRANSCRIPTION_COL = "Transcription"
BODY_COL = "Body"
INDEX_COL = "Index"


def extract_all_strings(obj):
    """
    Recursively extract all string leaves from a JSON-like structure.
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from extract_all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from extract_all_strings(item)
    else:
        return


def is_valid_transcription(value):
    if pd.isna(value):
        return False

    s = str(value).strip()
    if not s: return False

    # Try JSON first
    try:
        data = json.loads(s)
        
        strings = list(extract_all_strings(data))
        if not strings: return False
        
        joined = " ".join(strings).strip()
        if joined.lower().startswith("error:"): return False
        
        # No errors found
        return True
    except Exception:
        # Not JSON
        return False


def is_deleted_body(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() == "[deleted]"


def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path, dtype=str)
    original_count = len(df)
    print(f"Loaded {original_count} rows from {input_path}")

    # Ensure the key columns exist
    for col in (TRANSCRIPTION_COL, BODY_COL):
        if col not in df.columns:
            raise KeyError(f"Bad CSV: Column '{col}' not found")

    # Filter out bad rows
    mask_valid_transcription = df[TRANSCRIPTION_COL].apply(is_valid_transcription)
    mask_not_deleted_body = ~df[BODY_COL].apply(is_deleted_body)

    df_clean = df[mask_valid_transcription & mask_not_deleted_body].copy()
    cleaned_count = len(df_clean)

    removed = original_count - cleaned_count
    print(f"Removed {removed} rows ({original_count} -> {cleaned_count})")

    # Renumber Index
    df_clean[INDEX_COL] = range(1, cleaned_count + 1)

    # Save
    df_clean.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Cleaned dataset written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path",
    )

    args = parser.parse_args()
    clean_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
