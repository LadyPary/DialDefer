import json
from pathlib import Path

# Hard-coded list of files and/or folders
PATHS_TO_PROCESS = [
]

# Hard-coded keys to redact
KEYS_TO_REDACT = ["title", "original_transcript", "original_body"]

REDACTION_TEXT = "[redacted]"


def redact_jsonl_in_place(file_path: Path, keys_to_redact: list[str]) -> None:
    updated_lines = []

    with file_path.open("r", encoding="utf-8") as infile:
        for line_number, original_line in enumerate(infile, start=1):
            # Preserve blank lines exactly
            if not original_line.strip():
                updated_lines.append(original_line)
                continue

            try:
                obj = json.loads(original_line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON in {file_path} on line {line_number}: {e}")
                updated_lines.append(original_line)
                continue

            if isinstance(obj, dict):
                for key in keys_to_redact:
                    if key in obj:
                        obj[key] = REDACTION_TEXT

            updated_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    with file_path.open("w", encoding="utf-8") as outfile:
        outfile.writelines(updated_lines)

    print(f"Redacted and overwrote: {file_path}")


def get_jsonl_files_from_path(path_str: str) -> list[Path]:
    path = Path(path_str)

    if not path.exists():
        print(f"Path not found: {path}")
        return []

    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            return [path]
        print(f"Skipping non-jsonl file: {path}")
        return []

    if path.is_dir():
        return sorted(
            file for file in path.iterdir()
            if file.is_file() and file.suffix.lower() == ".jsonl"
        )

    print(f"Skipping unsupported path: {path}")
    return []


def main() -> None:
    files_to_process = []
    seen = set()

    for path_str in PATHS_TO_PROCESS:
        for file_path in get_jsonl_files_from_path(path_str):
            resolved = file_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files_to_process.append(file_path)

    if not files_to_process:
        print("No JSONL files found to process.")
        return

    for file_path in files_to_process:
        try:
            redact_jsonl_in_place(file_path, KEYS_TO_REDACT)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()