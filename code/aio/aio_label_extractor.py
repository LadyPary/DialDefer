#!/usr/bin/env python3
"""
Extract OR/NOR labels from AIO Reddit comments - IMPROVED VERSION.

Fixes:
1. Better pattern matching for negation ("isn't an overreaction", "not an overreaction")
2. "Underreacting" detection → NOR
3. Improved LLM prompt that clarifies:
   - Criticizing the OTHER person = supporting OP = NOR
   - Sympathizing with OP = NOR
   - Only OR if explicitly saying OP is being excessive

Usage:
    python aio_label_extractor_v2.py --input AIODataVerified.csv --output aio_labeled.jsonl --model openai/gpt-4o-mini
"""

import os
import re
import json
import argparse
import csv
import ast
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict, field

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from dotenv import load_dotenv


@dataclass
class LabelResult:
    """Result of label extraction."""
    label: Optional[str]  # "OR", "NOR", or None
    confidence: Optional[float]  # 0-1 for LLM, None for pattern matches
    method: str          # "pattern", "llm", or "failed"
    comment_index: int = 0  # Which comment in the list was used (0, 1, 2...)
    pattern_matched: Optional[str] = None  # The pattern that matched (if pattern method)
    reasoning: Optional[str] = None  # LLM reasoning (if llm method)
    source_comment: str = ""  # The actual comment that was labeled


# ---
# IMPROVED PATTERN MATCHING
# ---

def pattern_match_label(comment: str) -> LabelResult:
    """
    IMPROVED pattern matching with better negation handling.
    
    Key improvements:
    1. Check for NEGATED overreacting patterns FIRST (these indicate NOR)
    2. Add "underreacting" detection (= NOR)
    3. More robust negation patterns
    
    Args:
        comment: The Reddit comment text
        
    Returns:
        LabelResult with label, confidence, and method
    """
    if not comment or not comment.strip():
        return LabelResult(
            label=None,
            confidence=None,
            method="pattern",
            pattern_matched="empty_comment",
            source_comment=comment or ""
        )
    
    comment_stripped = comment.strip()
    comment_lower = comment_stripped.lower()
    
    # =====================================================================
    # NEGATION PATTERNS - CHECK THESE FIRST! (All indicate NOR)
    # =====================================================================
    
    # "not overreacting" / "not over reacting" / "not over-reacting"
    if re.search(r'\bnot\s+over[\s-]?reacting\b', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="not_overreacting",
            source_comment=comment_stripped
        )
    
    # "isn't overreacting" / "isnt overreacting" / "is not overreacting"
    if re.search(r'\b(isn\'?t|is\s+not)\s+(an?\s+)?over[\s-]?react', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="isnt_overreacting",
            source_comment=comment_stripped
        )
    
    # "aren't overreacting" / "are not overreacting"
    if re.search(r'\b(aren\'?t|are\s+not)\s+over[\s-]?react', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="arent_overreacting",
            source_comment=comment_stripped
        )
    
    # "don't think you're overreacting" / "dont think youre overreacting"
    if re.search(r'\bdon\'?t\s+think\s+(you\'?re|you\s+are|op\s+is)\s+over[\s-]?react', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="dont_think_overreacting",
            source_comment=comment_stripped
        )
    
    # "wouldn't say overreacting" / "wouldn't call this overreacting"
    if re.search(r'\b(wouldn\'?t|would\s+not)\s+(say|call)\s+(this\s+|it\s+)?(an?\s+)?over[\s-]?react', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="wouldnt_call_overreacting",
            source_comment=comment_stripped
        )
    
    # "this isn't an overreaction" / "that's not an overreaction"
    if re.search(r'\b(this|that|it)\s+(isn\'?t|is\s*n?o?t)\s+(an?\s+)?over[\s-]?reaction', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="this_isnt_overreaction",
            source_comment=comment_stripped
        )
    
    # "no overreaction" / "not an overreaction"
    if re.search(r'\b(no|not\s+an?)\s+over[\s-]?reaction\b', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="no_overreaction",
            source_comment=comment_stripped
        )
    
    # =====================================================================
    # UNDERREACTING PATTERNS (All indicate NOR - OP should react MORE)
    # =====================================================================
    
    # "underreacting" / "under reacting" / "under-reacting"
    if re.search(r'\bunder[\s-]?reacting\b', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="underreacting",
            source_comment=comment_stripped
        )
    
    # "underreaction" / "under reaction"
    if re.search(r'\bunder[\s-]?reaction\b', comment_lower):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="underreaction",
            source_comment=comment_stripped
        )
    
    # =====================================================================
    # EXPLICIT NOR MARKERS (Case-Sensitive for acronym)
    # =====================================================================
    
    # NOR at the very start: "NOR", "NOR.", "NOR,", "NOR:", "NOR -", "NOR!"
    if re.match(r'^NOR\s*[.,;:!\-]?\s', comment_stripped):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="NOR_start",
            source_comment=comment_stripped
        )
    
    # "NOR" as standalone at start (just "NOR" or "NOR.")
    if re.match(r'^NOR[.\s]*$', comment_stripped[:10]):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="NOR_standalone",
            source_comment=comment_stripped
        )
    
    # "You're NOR" or "you're NOR" - common pattern (NOR must be uppercase)
    if re.search(r"[Yy]ou'?re\s+NOR\b", comment_stripped):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="youre_NOR",
            source_comment=comment_stripped
        )
    
    # NOR at end of comment (common pattern: explanation then "NOR" or "NOR.")
    if re.search(r'\bNOR[.!]?\s*$', comment_stripped):
        return LabelResult(
            label="NOR",
            confidence=None,
            method="pattern",
            pattern_matched="NOR_end",
            source_comment=comment_stripped
        )
    
    # =====================================================================
    # EXPLICIT OR MARKERS (Case-Sensitive for acronym)
    # =====================================================================
    
    # OR at the very start: "OR", "OR.", "OR,", "OR:", "OR -"
    if re.match(r'^OR\s*[.,;:!\-]?\s', comment_stripped):
        return LabelResult(
            label="OR",
            confidence=None,
            method="pattern",
            pattern_matched="OR_start",
            source_comment=comment_stripped
        )
    
    # "OR" as standalone at start
    if re.match(r'^OR[.\s]*$', comment_stripped[:10]):
        return LabelResult(
            label="OR",
            confidence=None,
            method="pattern",
            pattern_matched="OR_standalone",
            source_comment=comment_stripped
        )
    
    # =====================================================================
    # EXPLICIT PHRASES (Case-Insensitive)
    # =====================================================================
    
    # "you're overreacting" or "you are overreacting" - clear OR signal
    # But NOT "you're not overreacting" (handled above)
    if re.search(r"\byou'?re\s+(definitely\s+|100%\s+)?over[\s-]?reacting\b", comment_lower):
        # Double-check no negation nearby
        if not re.search(r'\b(not|aren\'?t|isn\'?t)\b.{0,10}over[\s-]?react', comment_lower):
            return LabelResult(
                label="OR",
                confidence=None,
                method="pattern",
                pattern_matched="youre_overreacting_phrase",
                source_comment=comment_stripped
            )
    
    # "100% overreacting" or "definitely overreacting"
    if re.search(r'\b(100%|definitely|absolutely|totally)\s+over[\s-]?reacting\b', comment_lower):
        return LabelResult(
            label="OR",
            confidence=None,
            method="pattern",
            pattern_matched="definitely_overreacting",
            source_comment=comment_stripped
        )
    
    # "this is an overreaction" / "that's an overreaction"
    if re.search(r'\b(this|that)\s+(is|was)\s+(an?\s+)?over[\s-]?reaction\b', comment_lower):
        # Make sure no negation
        if not re.search(r'\b(not|no|isn\'?t)\b.{0,10}over[\s-]?reaction', comment_lower):
            return LabelResult(
                label="OR",
                confidence=None,
                method="pattern",
                pattern_matched="this_is_overreaction",
                source_comment=comment_stripped
            )
    
    # No pattern matched - needs LLM
    return LabelResult(
        label=None,
        confidence=None,
        method="pattern",
        pattern_matched="no_match",
        source_comment=comment_stripped
    )


def parse_comment_list(comment_field: str) -> List[str]:
    """
    Parse a comment field that might be a JSON list or Python list string.
    """
    if not comment_field or comment_field.strip() == "":
        return []
    
    # Try JSON parsing first
    try:
        comments = json.loads(comment_field)
        if isinstance(comments, list):
            return [str(c) for c in comments if c]
        return [str(comments)] if comments else []
    except json.JSONDecodeError:
        pass
    
    # Try Python literal eval
    try:
        comments = ast.literal_eval(comment_field)
        if isinstance(comments, list):
            return [str(c) for c in comments if c]
        return [str(comments)] if comments else []
    except (ValueError, SyntaxError):
        pass
    
    # Fallback: treat as single comment
    return [comment_field] if comment_field.strip() else []


def extract_label_with_pattern_fallback(
    comments: List[str],
    max_attempts: int = 3
) -> LabelResult:
    """
    Try to extract label from a list of comments using pattern matching.
    Falls back through the list if no pattern matches.
    """
    for i, comment in enumerate(comments[:max_attempts]):
        result = pattern_match_label(comment)
        result.comment_index = i
        
        # If we got a label, return it
        if result.label is not None:
            return result
    
    # No label found in any comment via pattern
    return LabelResult(
        label=None,
        confidence=None,
        method="pattern",
        comment_index=-1,
        pattern_matched="all_no_match",
        source_comment=comments[0] if comments else ""
    )


# ---
# IMPROVED LLM EXTRACTION
# ---

def get_label_extraction_prompt(title: str, body: str, comment: str) -> str:
    """
    IMPROVED prompt to extract OR/NOR label from a Reddit comment.
    
    Key improvements:
    1. Clearer explanation that criticizing OTHER person = supporting OP = NOR
    2. Examples of common misinterpretation cases
    3. Emphasis on WHO the commenter is criticizing
    """
    # Truncate body if too long (keep first 800 chars for context)
    body_truncated = body[:800] + "..." if len(body) > 800 else body
    
    return f"""You are analyzing a Reddit comment from r/AmIOverreacting.

In this subreddit, the Original Poster (OP) describes a situation where they had a reaction to something, and asks if they are overreacting. Commenters judge whether OP's reaction was justified.

**POST TITLE:** "{title}"

**POST BODY (OP's situation):**
{body_truncated}

**COMMENT TO ANALYZE:**
"{comment}"

## CLASSIFICATION RULES:

**NOR (Not Overreacting)** - The commenter thinks OP's reaction is JUSTIFIED. Signs include:
- Commenter criticizes the OTHER person in OP's story (boyfriend, friend, coworker, etc.)
- Commenter sympathizes with OP or validates their feelings
- Commenter says things like "red flag", "that's messed up", "you deserve better"
- Commenter says OP is "underreacting" (meaning OP should be MORE upset)
- Commenter defends OP's position or actions

**OR (Overreacting)** - The commenter thinks OP's reaction is EXCESSIVE. Signs include:
- Commenter directly tells OP they are being dramatic, excessive, or making a big deal
- Commenter defends the OTHER person's behavior as normal/acceptable
- Commenter says OP needs to calm down or let it go
- Commenter minimizes what happened to OP

## CRITICAL DISTINCTION:
If the commenter is attacking/criticizing the OTHER person (not OP), this means they are SIDING WITH OP, which is **NOR**.

Example: If OP complains about their boyfriend, and the commenter says "your boyfriend is a jerk" or "he's gaslighting you" - this is **NOR** because they're supporting OP.

Return ONLY a JSON object:
{{"label": "OR" or "NOR" or "UNCLEAR", "confidence": 0.0-1.0, "reasoning": "one sentence explaining WHO the commenter is criticizing"}}

JSON:"""


def llm_extract_label(
    title: str,
    body: str,
    comment: str,
    client,
    model: str,
    temperature: float = 0.0
) -> LabelResult:
    """
    Use LLM to extract label with improved prompt.
    """
    prompt = get_label_extraction_prompt(title, body, comment)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200
        )
        
        reply = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if "```" in reply:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', reply, re.DOTALL)
            if match:
                reply = match.group(1)
        
        # Parse JSON
        data = json.loads(reply)
        label = data.get("label", "UNCLEAR")
        confidence = float(data.get("confidence", 0.5))
        
        if label == "UNCLEAR":
            label = None
        
        return LabelResult(
            label=label,
            confidence=confidence,
            method="llm",
            pattern_matched=None,
            reasoning=data.get("reasoning", ""),
            source_comment=comment
        )
        
    except Exception as e:
        print(f"    LLM extraction failed: {e}")
        return LabelResult(
            label=None,
            confidence=None,
            method="llm_failed",
            pattern_matched=None,
            reasoning=f"error:{str(e)[:100]}",
            source_comment=comment
        )


def llm_extract_with_fallback(
    title: str,
    body: str,
    comments: List[str],
    client,
    model: str,
    max_attempts: int = 3,
    temperature: float = 0.0
) -> LabelResult:
    """
    Try LLM extraction on comments, falling back through the list.
    """
    for i, comment in enumerate(comments[:max_attempts]):
        result = llm_extract_label(title, body, comment, client, model, temperature)
        result.comment_index = i
        
        if result.label is not None:
            return result
    
    # All comments were UNCLEAR
    return LabelResult(
        label=None,
        confidence=None,
        method="llm",
        comment_index=-1,
        pattern_matched=None,
        reasoning="all_unclear",
        source_comment=comments[0] if comments else ""
    )


# ---
# MAIN EXTRACTION PIPELINE
# ---

def extract_label_for_record(
    record: Dict[str, Any],
    use_llm: bool = True,
    client = None,
    model: str = None
) -> Dict[str, Any]:
    """
    Extract OR/NOR label for a single record.
    
    Pipeline:
    1. Try pattern matching on BestComments (fast, high-confidence)
    2. If no match and LLM enabled, try LLM on BestComments with title+body context
    3. If still no match, try TopComments
    """
    result = dict(record)
    
    title = record.get("Title", "")
    body = record.get("Body", "")
    best_comments = parse_comment_list(record.get("BestComments", ""))
    top_comments = parse_comment_list(record.get("TopComments", ""))
    
    # Step 1: Try pattern matching on BestComments
    label_result = extract_label_with_pattern_fallback(best_comments, max_attempts=3)
    
    # Step 2: If pattern failed and LLM enabled, try LLM on BestComments
    if label_result.label is None and use_llm and client and best_comments:
        label_result = llm_extract_with_fallback(
            title, body, best_comments, client, model, max_attempts=3
        )
    
    # Step 3: If still no label, try pattern matching on TopComments
    if label_result.label is None and top_comments:
        top_result = extract_label_with_pattern_fallback(top_comments, max_attempts=3)
        if top_result.label is not None:
            top_result.method = "pattern_top"
            label_result = top_result
    
    # Step 4: If still no label and LLM enabled, try LLM on TopComments
    if label_result.label is None and use_llm and client and top_comments:
        top_result = llm_extract_with_fallback(
            title, body, top_comments, client, model, max_attempts=3
        )
        if top_result.label is not None:
            top_result.method = "llm_top"
            label_result = top_result
    
    # Store results
    result["ground_truth"] = label_result.label
    result["label_confidence"] = label_result.confidence
    result["label_method"] = label_result.method
    result["label_comment_index"] = label_result.comment_index
    result["label_pattern"] = label_result.pattern_matched  # Only set for pattern matches
    result["label_reasoning"] = label_result.reasoning  # Only set for LLM
    result["label_source_comment"] = label_result.source_comment[:500]
    
    return result


def process_csv(
    input_file: str,
    output_file: str,
    use_llm: bool = True,
    llm_model: str = "openai/gpt-4o-mini",
    api_key: str = None,
    max_rows: int = None,
    temperature: float = 0.0
) -> Dict[str, int]:
    """
    Process entire CSV file and extract labels.
    """
    load_dotenv()
    
    # Setup LLM client if needed
    client = None
    if use_llm:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            print(f"LLM client initialized (model: {llm_model})")
        except ImportError:
            print("Warning: openai package not installed, LLM fallback disabled")
            use_llm = False
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            use_llm = False
    
    # Read CSV
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    if max_rows:
        records = records[:max_rows]
    
    print(f"Loaded {len(records)} records")
    print(f"  LLM enabled: {use_llm}")
    
    # Process records
    stats = {
        "total": len(records),
        "OR": 0,
        "NOR": 0,
        "unlabeled": 0,
        "pattern": 0,
        "pattern_top": 0,
        "llm": 0,
        "llm_top": 0,
    }
    
    # Clear output file
    open(output_file, 'w').close()
    
    iterator = tqdm(records, desc="Extracting labels") if HAS_TQDM else records
    
    for i, record in enumerate(iterator):
        result = extract_label_for_record(
            record,
            use_llm=use_llm,
            client=client,
            model=llm_model
        )
        
        # Update stats
        gt = result["ground_truth"]
        if gt == "OR":
            stats["OR"] += 1
        elif gt == "NOR":
            stats["NOR"] += 1
        else:
            stats["unlabeled"] += 1
        
        method = result["label_method"]
        if method in stats:
            stats[method] += 1
        
        # Write to JSONL
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        if not HAS_TQDM and (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(records)} | OR:{stats['OR']} NOR:{stats['NOR']} unlabeled:{stats['unlabeled']}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("LABEL EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total records:     {stats['total']}")
    print(f"  OR (overreact):    {stats['OR']} ({stats['OR']/max(1,stats['total'])*100:.1f}%)")
    print(f"  NOR (not overr.):  {stats['NOR']} ({stats['NOR']/max(1,stats['total'])*100:.1f}%)")
    print(f"  Unlabeled:         {stats['unlabeled']} ({stats['unlabeled']/max(1,stats['total'])*100:.1f}%)")
    print("-" * 60)
    print("  By method:")
    print(f"    Pattern (best):  {stats['pattern']}")
    print(f"    Pattern (top):   {stats['pattern_top']}")
    if use_llm:
        print(f"    LLM (best):      {stats['llm']}")
        print(f"    LLM (top):       {stats['llm_top']}")
    print("=" * 60)
    print(f"\nOutput saved to {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract OR/NOR labels from AIO Reddit comments (IMPROVED VERSION)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With LLM (recommended)
  python aio_label_extractor_v2.py --input AIOData.csv --output aio_labeled.jsonl --model openai/gpt-4o-mini
  
  # Pattern matching only (faster but more unlabeled)
  python aio_label_extractor_v2.py --input AIOData.csv --output aio_labeled.jsonl --no-llm
  
  # Test on first 20 rows
  python aio_label_extractor_v2.py --input AIOData.csv --output aio_labeled.jsonl --model openai/gpt-4o-mini --max-rows 20
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--model", "-m", default="openai/gpt-4o-mini", help="LLM model for fallback")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback (pattern only)")
    parser.add_argument("--api-key", help="API key (uses OPENROUTER_API_KEY env var if not set)")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to process")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    
    args = parser.parse_args()
    
    process_csv(
        input_file=args.input,
        output_file=args.output,
        use_llm=not args.no_llm,
        llm_model=args.model,
        api_key=args.api_key,
        max_rows=args.max_rows,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()