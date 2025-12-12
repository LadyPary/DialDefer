#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASET_PATH = "AIODataFullClean.csv"
TRANSCRIPTION_COL = "Transcription"

SCORE_MAX = 15000
SCORE_BIN_WIDTH = 1000


# Parse transcriptions
def parse_transcription_messages(transcription_value):
    if pd.isna(transcription_value):
        return []

    s = str(transcription_value).strip()
    if not s:
        return []

    try:
        data = json.loads(s)
    except Exception:
        return []

    if isinstance(data, dict):
        msgs = data.get("messages", [])
        if isinstance(msgs, list):
            return msgs
        return []


    return []

def get_message_text(m):
    if not isinstance(m, dict):
        return ""
    text = m.get("text")
    return text if isinstance(text, str) else ""


# Conversatoin Length (Simple Word Count)
def compute_conversation_length(transcription_value):
    msgs = parse_transcription_messages(transcription_value)
    if msgs:
        text_parts = [get_message_text(m) for m in msgs if get_message_text(m)]
        if text_parts:
            return len(" ".join(text_parts).split())

    if pd.isna(transcription_value):
        return 0
    s = str(transcription_value).strip()
    return len(s.split()) if s else 0

# Turns (How many times does the speaker change)
def compute_turns_for_conversation(transcription_value):
    msgs = parse_transcription_messages(transcription_value)
    if not msgs:
        return 0

    speakers = []
    for m in msgs:
        sp = m.get("speaker")
        if isinstance(sp, str) and sp.strip():
            speakers.append(sp.strip())

    if len(speakers) < 2:
        return 0

    turns = 0
    prev = speakers[0]
    for curr in speakers[1:]:
        if curr != prev:
            turns += 1
        prev = curr

    return turns

# Helper to get the word count for each spekaer
def compute_speaker_word_counts(transcription_value):
    msgs = parse_transcription_messages(transcription_value)
    if not msgs:
        return {}

    counts = {}
    for m in msgs:
        sp = m.get("speaker")
        msg = get_message_text(m)
        if not isinstance(sp, str) or not isinstance(msg, str):
            continue
        sp = sp.strip()
        if not sp:
            continue
        wc = len(msg.split())
        counts[sp] = counts.get(sp, 0) + wc

    return counts

# Speaker Domination (What percentage of the conversation is one speaker by word count)
def compute_speaker_a_percentage(transcription_value):
    counts = compute_speaker_word_counts(transcription_value)
    if not counts:
        return None

    total = sum(counts.values())
    if total == 0:
        return None

    a_words = 0
    for sp, wc in counts.items():
        if sp.lower() == "speaker a":
            a_words += wc

    return 100.0 * a_words / total

# Intelligently choose a width for the bars on histogram
def choose_bin_width(values, target_bins=20):
    if not values:
        return 10
    max_val = max(values)
    raw_width = max_val / max(target_bins, 1)
    width = int(math.ceil(raw_width / 10.0) * 10)
    return max(width, 10)

# Remove outliers on upper and lower ends
def compute_trimmed_mean(values, trim_percent):
    if not values:
        return 0.0
    n = len(values)
    if trim_percent <= 0:
        return sum(values) / n

    vals = sorted(values)
    k = int(n * (trim_percent / 100.0))

    if 2 * k >= n:
        return sum(vals) / n

    mid = vals[k:n - k]
    return sum(mid) / len(mid) if mid else 0.0



def analyze_and_plot(csv_path, save_all=False, trim_percent=5.0, tick_fontsize=5):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, dtype=str)

    if TRANSCRIPTION_COL not in df.columns:
        raise KeyError(f"Missing '{TRANSCRIPTION_COL}' column.")

    # Print to terminal and to save to txt file later
    stats_lines = []
    def log(line=""):
        print(line)
        stats_lines.append(line)

    log(f"Loaded {len(df)} rows from {csv_path}\n")

    lengths = []
    turns = []
    domination = []

    for val in df[TRANSCRIPTION_COL]:
        lengths.append(compute_conversation_length(val))
        turns.append(compute_turns_for_conversation(val))
        pct = compute_speaker_a_percentage(val)
        if pct is not None:
            domination.append(pct)

    #Conversation Length Stats
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    trimmed_avg_len = compute_trimmed_mean(lengths, trim_percent)

    log("=== Conversation Length Stats ===")
    log(f"Conversations: {len(lengths)}")
    log(f"Min length: {min_len} words")
    log(f"Max length: {max_len} words")
    log(f"Average length (all data): {avg_len:.2f} words")
    log(
        f"Trimmed average length (removing bottom {trim_percent:.1f}% and top {trim_percent:.1f}%): "
        f"{trimmed_avg_len:.2f} words"
    )
    log()

    # Speaker Turn Stats
    max_turns = max(turns)
    min_turns = min(turns)
    avg_turns = sum(turns) / len(turns)

    log("=== Speaker Turn Stats ===")
    log(f"Min turns: {min_turns}")
    log(f"Max turns: {max_turns}")
    log(f"Average turns (all data): {avg_turns:.2f}")
    log()

    # Speaker Domination Stats
    if domination:
        min_dom = min(domination)
        max_dom = max(domination)
        avg_dom = sum(domination) / len(domination)
        balanced = sum(1 for d in domination if 40.0 <= d <= 60.0)

        log("=== Speaker Domination Stats ===")
        log(f"Conversations counted: {len(domination)}")
        log(f"Min % words from Speaker A: {min_dom:.2f}%")
        log(f"Max % words from Speaker A: {max_dom:.2f}%")
        log(f"Average % words from Speaker A: {avg_dom:.2f}%")
        log(f"Conversations ~balanced (40–60% for Speaker A): {balanced}")
    else:
        log("=== Speaker Domination Stats (Speaker A) ===")
        log("No conversations with measurable Speaker A percentage.")
    log()

    # Score Stats
    log("=== Score Stats ===")
    raw_scores = []
    if "Score" not in df.columns:
        log("No 'Score' column found — skipping score stats and histogram.")
    else:
        for x in df["Score"]:
            try:
                raw_scores.append(int(x))
            except Exception:
                pass

        if not raw_scores:
            log("No valid numeric scores found — skipping score histogram.")
        else:
            max_score = max(raw_scores)
            min_score = min(raw_scores)
            avg_score = sum(raw_scores) / len(raw_scores)
            trimmed_avg_score = compute_trimmed_mean(raw_scores, trim_percent)

            log(f"Valid scores: {len(raw_scores)}")
            log(f"Min score: {min_score}")
            log(f"Max score: {max_score}")
            log(f"Average score (all data): {avg_score:.2f}")
            log(f"Trimmed average score (removing bottom {trim_percent:.1f}% and top {trim_percent:.1f}%): {trimmed_avg_score:.2f}")
    log()

    # Issues with matplotlib not printing
    sys.stdout.flush()

    # Histogram generation with help of GPT:

    # ----------- Conversation Length Histogram -----------
    bw = choose_bin_width(lengths)

    upper = ((max_len // bw) + 1) * bw
    bins = np.arange(0, upper + bw, bw)

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins, edgecolor="black")
    plt.title("Distribution of Conversation Lengths")
    plt.xlabel("Total word count")
    plt.ylabel("Number of conversations")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(bins[1:], [str(int(b)) for b in bins[1:]], fontsize=tick_fontsize)

    if save_all:
        os.makedirs("output", exist_ok=True)
        plt.savefig("output/conversation_lengths.png", bbox_inches="tight")
        print("Saved: output/conversation_lengths.png")

    # ----------- Turn Histogram -----------
    bins_turns = np.arange(-0.5, max_turns + 1.5, 1)

    plt.figure(figsize=(10, 6))
    plt.hist(turns, bins=bins_turns, edgecolor="black")
    plt.title("Distribution of Turns")
    plt.xlabel("Turns (speaker swaps)")
    plt.ylabel("Number of conversations")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(range(0, max_turns + 1), fontsize=tick_fontsize)

    if save_all:
        plt.savefig("output/turns.png", bbox_inches="tight")
        print("Saved: output/turns.png")

    # ----------- Speaker Domination Histogram -----------
    if domination:
        bins_dom = np.arange(0, 105, 5)

        plt.figure(figsize=(10, 6))
        plt.hist(domination, bins=bins_dom, edgecolor="black")
        plt.title("Distribution of Speaker A Domination (%)")
        plt.xlabel("Speaker A share of words (%)")
        plt.ylabel("Number of conversations")
        plt.grid(axis="y", alpha=0.3)
        plt.xticks(bins_dom[1:], [f"{int(b)}%" for b in bins_dom[1:]], fontsize=tick_fontsize)

        if save_all:
            plt.savefig("output/speaker_domination.png", bbox_inches="tight")
            print("Saved: output/speaker_domination.png")

    # ----------- Score Histogram (fixed bins, 15000, overflow) -----------
    if raw_scores:
        scores = [s if s <= SCORE_MAX else SCORE_MAX + 1 for s in raw_scores]

        bins_score = np.arange(0, SCORE_MAX + SCORE_BIN_WIDTH * 2, SCORE_BIN_WIDTH)

        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=bins_score, edgecolor="black")
        plt.title("Distribution of Scores (15000+ overflow)")
        plt.xlabel("Score")
        plt.ylabel("Number of conversations")
        plt.grid(axis="y", alpha=0.3)

        labels = []
        for i, b in enumerate(bins_score[1:]):
            if i == len(bins_score) - 2:
                labels.append("15001+")
            else:
                labels.append(str(int(b)))

        plt.xticks(bins_score[1:], labels, fontsize=tick_fontsize)

        if save_all:
            plt.savefig("output/scores.png", bbox_inches="tight")
            print("Saved: output/scores.png")

    # ----------- Save stats.txt if requested -----------
    if save_all:
        os.makedirs("output", exist_ok=True)
        stats_path = os.path.join("output", "stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("\n".join(stats_lines))
        print(f"Saved: {stats_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate dataset statistics.")
    parser.add_argument("--csv", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--trim-percent", type=float, default=5.0)

    args = parser.parse_args()

    analyze_and_plot(
        csv_path=args.csv,
        save_all=args.save,
        trim_percent=args.trim_percent,
    )


if __name__ == "__main__":
    main()
