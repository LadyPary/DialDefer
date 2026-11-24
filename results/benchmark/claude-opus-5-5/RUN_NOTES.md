# Claude Opus 5.5 — DialDefer main-table run

**Run date:** 2026-11-24 · **Model:** `claude-opus-5-5` (Anthropic API, Python SDK `anthropic` 2.13.0)

## Settings (differ from the earlier models — note in paper)
- Effort `medium` (the API default for this model). Thinking is always on and cannot be disabled.
- No `temperature`/`seed`: the API rejects sampling params on this model, and it has no seed.
- `max_tokens` 16000 (thinking counts toward it; the 300-token cap used for other models would truncate).
- Prompts, datasets, conditions and speaker labels are unchanged (`Speaker 1`/`Speaker 2`; r/AIO: `Speaker B judges Speaker A` main, `Speaker A judges Speaker B` with C1+C2 as for GPT-5-mini).
- No server-side refusal fallback (a fallback would answer with a different model).

## Run stats
- 3,244 items (2,964 benchmark + 280 r/AIO); 14,134 API calls; cost ≈ $103.
- 36 refusals (`stop_reason: refusal`), reconciled against the outputs:
  - AdvisorQA: 18 `cyber` refusals → 18 empty replies (5 items) → missing answers.
  - GPQA: 18 `bio` refusals (6 items) → 10 empty + 8 cut-off replies.
  - No other dataset and no r/AIO call was refused; 0 responses hit `max_tokens`.
- Missing answers are scored as wrong (same as for the other models). The analyzer's JSON recovery
  restored 15 of 17 unparsed replies: 9 were complete JSON whose LaTeX braces break
  `extract_json_from_response`, and 6 were GPQA replies cut off by a refusal after the answer field.
- Sensitivity:
  - AdvisorQA, excluding refused items: +24.7 → +24.8.
  - GPQA: analyzer default −1.5; refused calls scored as wrong +0.7; refused items excluded +0.8 (N=128).
  - All are non-significant, and the 10-domain mean moves by at most +0.2.

## DDS (pp)
`***` = the 95% bootstrap CI of DDS excludes 0 (`DDS_Significant` in bench_analyzer). It is **not** a p-value.
McNemar p-values per condition pair are in the summary CSV (e.g. BBQ: p=.06 / 1.0, not significant).

| Domain | DDS | 95% CI |
|---|---|---|
| AdvisorQA | +24.7*** | [+17.7, +32.0] |
| AMQA | +0.8 | [−2.1, +3.8] |
| BBQ | −1.7*** | [−3.3, −0.3] |
| GPQA | −1.5 | [−9.0, +5.2] |
| HaluEval | +3.3 | [−0.3, +7.3] |
| HARP | +0.0 | [0.0, 0.0] |
| PlausibleQA | +2.0 | [−1.3, +5.7] |
| SocialIQA | +6.0*** | [+1.7, +10.3] |
| TruthfulQA | +4.1*** | [+2.5, +5.7] |
| r/AIO (B judges A) | +31.4 | (aio_analyzer reports no CI) |
| **Mean over 10 domains** (paper metric; reproduces Qwen +33.8, Gemma +29.5, GPT-4o −1.1) | **+6.9** | |

Mean accuracy over the 10 domains: 82.5% (C1 82.9%, C2 82.1%). r/AIO "A judges B": DDS +2.5.

## Files
- `claude-opus-5-5_analysis_summary.csv`, `claude-opus-5-5_analysis_summary_with_aio.csv` (this folder)
- `results/aio/analysis/{b judges a (main),a judges b}/aio_claude-opus-5-5_*_analysis.csv`
- Raw outputs: `*_all.jsonl` here and `results/aio/results/claude-opus-5-5/`; per-call log `results/_claude_calls.jsonl` (gitignored)
