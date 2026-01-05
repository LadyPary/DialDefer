#!/usr/bin/env python3
"""
Generate Reports from Taxonomy JSON Output
==========================================

Takes the JSON output from run_full_taxonomy.py and generates:
1. Comprehensive Markdown report with tables and statistics
2. LaTeX tables for academic papers

Usage:
    python generate_reports.py taxonomy_output.json
    python generate_reports.py taxonomy_output.json -o custom_report
    python generate_reports.py taxonomy_output.json --latex-only
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any


# Code descriptions for reports
CODE_DESCRIPTIONS = {
    # Social Attribution
    'SA1': 'Speaker Authority - treats speaker as inherently authoritative',
    'SA2': 'Dialogue Deference - defers to conversational format',
    'SA3': 'Validation Seeking - validates for social harmony',
    'SA4': 'Role-Based Trust - trusts based on role',
    
    # Evidential Standards
    'ES1': 'Standard Lowering - lowers evidence requirements',
    'ES2': 'Sufficiency Shift - same info treated as sufficient',
    'ES3': 'False Grounding - claims nonexistent evidence',
    'ES4': 'Selective Reading - cherry-picks dialogue',
    
    # Epistemic Shift
    'EP1': 'Uncertainty Collapse - certainty from uncertainty',
    'EP2': 'Hedging Asymmetry - drops hedges in one condition',
    'EP3': 'Verification Refusal - refuses to confirm without re-verification',
    
    # Reasoning Error
    'RE1': 'Circular Reasoning - claim as its own evidence',
    'RE2': 'Factual Contradiction - contradicts known facts',
    'RE3': 'Calculation Error - arithmetic/domain errors',
    'RE4': 'Comprehension Error - misreads question/answer',
    
    # Internal Incoherence
    'IC1': 'Reasoning-Answer Mismatch - reasoning supports opposite',
    'IC2': 'Self-Contradiction - contradictory claims in reasoning',
    'IC3': 'Acknowledged Incorrectness - notes wrong but chooses anyway',
    
    # Knowledge Inconsistency
    'KI1': 'Belief Shift - model belief changes between conditions',
    'KI2': 'Knowledge Type Shift - different knowledge types used',
    'KI3': 'Factual Inconsistency - cites different facts',
    
    # Evaluation Criteria
    'EV1': 'Strictness Asymmetry - stricter in one condition',
    'EV2': 'Completeness Standard - requires more in one condition',
    'EV3': 'Literal vs Pragmatic - interpretation shift',
    'EV4': 'Scope Difference - different question scope',
    'EV5': 'Normative Injection - applies ethics/legal in one only',
    'EV6': 'Genre Mismatch - rejects humor/abstention as invalid',
    'EV7': 'Approximation Tolerance - accepts close enough in one',
    
    # Emergent
    'EM1': 'Novel Pattern - pattern not in taxonomy',
    
    # Aliases from older taxonomy
    'ER1': 'Standard Lowering',
    'ER2': 'Sufficiency Shift',
    'ER3': 'False Grounding',
    'ER4': 'Selective Reading',
    'EC1': 'Uncertainty Collapse',
    'EC2': 'Hedging Removal',
    'EC3': 'Unknown Rejection',
    'EC4': 'Overconfidence Injection',
    'LF1': 'Circular Reasoning',
    'LF2': 'Factual Override',
    'LF3': 'Non Sequitur',
    'LF4': 'Format Conflation',
    'IC4': 'Acknowledged Error',
    'EP1_OLD': 'Novel Pattern',
    'EP2_OLD': 'Domain-Specific Pattern',
    'EP3_OLD': 'Framing Effect',
}

CATEGORY_NAMES = {
    'SOCIAL_ATTRIBUTION': 'Social Attribution',
    'EVIDENTIAL_STANDARDS': 'Evidential Standards',
    'EVIDENTIAL_RELAXATION': 'Evidential Relaxation',
    'EPISTEMIC_SHIFT': 'Epistemic Shift',
    'EPISTEMIC_COLLAPSE': 'Epistemic Collapse',
    'REASONING_ERROR': 'Reasoning Error',
    'LOGICAL_FAILURE': 'Logical Failure',
    'INTERNAL_INCOHERENCE': 'Internal Incoherence',
    'KNOWLEDGE_INCONSISTENCY': 'Knowledge Inconsistency',
    'EVALUATION_CRITERIA': 'Evaluation Criteria',
    'EMERGENT_PATTERN': 'Emergent Pattern',
}


def generate_markdown_report(data: Dict) -> str:
    """Generate comprehensive markdown report from taxonomy JSON."""
    
    lines = [
        "# DialDefer Taxonomy Analysis Report",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Metadata
    meta = data.get('metadata', {})
    lines.extend([
        f"**Input Folders**: {', '.join(meta.get('input_folders', ['N/A']))}",
        f"**Total Records**: {meta.get('total_records', 'N/A')}",
        f"**Total Flips**: {meta.get('total_flips', 'N/A')}",
        f"**Models Analyzed**: {', '.join(meta.get('models_analyzed', ['N/A']))}",
        "",
    ])
    
    # Extraction stats
    ext = data.get('extraction_stats', {})
    if ext:
        lines.extend([
            "### Flip Extraction Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Deference Flips | {ext.get('deference_flips', 0)} |",
            f"| Skepticism Flips | {ext.get('skepticism_flips', 0)} |",
            f"| Total Flips | {ext.get('deference_flips', 0) + ext.get('skepticism_flips', 0)} |",
            "",
        ])
        
        # By model
        by_model = ext.get('by_model', {})
        if by_model:
            lines.extend([
                "#### By Model",
                "",
                "| Model | Records | Deference | Skepticism | Total Flips |",
                "|-------|---------|-----------|------------|-------------|",
            ])
            for model, stats in by_model.items():
                lines.append(
                    f"| {model} | {stats.get('records', 0)} | "
                    f"{stats.get('deference', 0)} | {stats.get('skepticism', 0)} | "
                    f"{stats.get('total_flips', 0)} |"
                )
            lines.append("")
        
        # By dataset
        by_ds = ext.get('by_dataset', {})
        if by_ds:
            lines.extend([
                "#### By Dataset",
                "",
                "| Dataset | Deference | Skepticism | Total |",
                "|---------|-----------|------------|-------|",
            ])
            for ds, stats in sorted(by_ds.items()):
                lines.append(
                    f"| {ds} | {stats.get('deference', 0)} | "
                    f"{stats.get('skepticism', 0)} | {stats.get('total', 0)} |"
                )
            lines.append("")
    
    # Quantitative Analysis
    quant = data.get('quantitative_analysis', {})
    if quant:
        lines.extend([
            "---",
            "",
            "## Quantitative Analysis",
            "",
        ])
        
        # Per-model stats
        per_model = quant.get('per_model', {})
        if per_model:
            lines.extend([
                "### Per-Model Summary",
                "",
                "| Model | Items | Flips | Flip Rate | DDS Mean | DDS Std |",
                "|-------|-------|-------|-----------|----------|---------|",
            ])
            for model, stats in per_model.items():
                lines.append(
                    f"| {model} | {stats.get('n_items', 0)} | "
                    f"{stats.get('n_any_flip', 0)} | {stats.get('flip_rate', 0):.1f}% | "
                    f"{stats.get('dds_mean', 0):.3f} | {stats.get('dds_std', 0):.3f} |"
                )
            lines.append("")
        
        # Domain stats by model
        domain_stats = quant.get('domain_stats_by_model', {})
        if domain_stats:
            for model, ds_stats in domain_stats.items():
                lines.extend([
                    f"### {model} - Domain Statistics",
                    "",
                    "| Dataset | N | C1 Acc | C2 Acc | Δ Correct | Δ Incorrect | DDS | Flip Rate |",
                    "|---------|---|--------|--------|-----------|-------------|-----|-----------|",
                ])
                for ds, s in sorted(ds_stats.items()):
                    lines.append(
                        f"| {ds} | {s.get('n_items', 0)} | "
                        f"{s.get('c1_avg_acc', 0):.1f}% | {s.get('c2_avg_acc', 0):.1f}% | "
                        f"{s.get('delta_correct', 0):+.1f} | {s.get('delta_incorrect', 0):+.1f} | "
                        f"{s.get('dds', 0):+.1f} | {s.get('flip_rate', 0):.1f}% |"
                    )
                lines.append("")
    
    # Cross-model analysis
    cross = data.get('cross_model_analysis', {})
    if cross and cross.get('total_common_items', 0) > 0:
        lines.extend([
            "---",
            "",
            "## Cross-Model Analysis",
            "",
            f"**Models Compared**: {', '.join(cross.get('models_analyzed', []))}",
            f"**Common Items**: {cross.get('total_common_items', 0)}",
            "",
        ])
        
        summary = cross.get('summary', {})
        if summary:
            lines.extend([
                "### Flip Consistency",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Items flipping in ALL models | {summary.get('items_flip_all_models', 0)} ({summary.get('universal_flip_rate', 0)}%) |",
                f"| Items flipping in NO models | {summary.get('items_flip_no_models', 0)} |",
                f"| Items flipping in SOME models | {summary.get('items_flip_some_models', 0)} |",
                f"| Agreement Rate | {summary.get('agreement_rate', 0)}% |",
                "",
            ])
        
        # DDS comparison
        dds_comp = cross.get('dds_comparison', {})
        if dds_comp:
            models = cross.get('models_analyzed', [])
            lines.extend([
                "### DDS Comparison Across Models",
                "",
                "| Dataset | " + " | ".join(models) + " | Consistent |",
                "|---------|" + "|".join(["------" for _ in models]) + "|------------|",
            ])
            for ds, ds_data in sorted(dds_comp.items()):
                dds_values = [f"{ds_data.get(m, {}).get('dds', 0):+.1f}" for m in models]
                consistent = "[ok]" if ds_data.get('_consistent', False) else "[x]"
                lines.append(f"| {ds} | " + " | ".join(dds_values) + f" | {consistent} |")
            lines.append("")
        
        # Consistently flipping items
        consistently_flipping = cross.get('consistently_flipping_items', [])
        if consistently_flipping:
            lines.extend([
                "### Items Flipping in ALL Models (Top 10)",
                "",
            ])
            for i, item in enumerate(consistently_flipping[:10], 1):
                lines.append(f"{i}. **[{item['dataset']}]** {item.get('question', 'N/A')[:80]}...")
            lines.append("")
    
    # LLM Taxonomy Results
    llm = data.get('llm_taxonomy', {})
    if llm:
        lines.extend([
            "---",
            "",
            "## LLM-as-Judge Taxonomy Analysis",
            "",
        ])
        
        llm_meta = llm.get('metadata', {})
        lines.extend([
            "### Analysis Metadata",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Flips Analyzed | {llm_meta.get('total_flips_analyzed', 0)} |",
            f"| Valid Analyses | {llm_meta.get('valid_analyses', 0)} |",
            f"| Errors | {llm_meta.get('errors', 0)} |",
            f"| Judge Model | {llm_meta.get('judge_model', 'N/A')} |",
            f"| Analysis Time | {llm_meta.get('analysis_time_seconds', 0):.1f}s |",
            "",
        ])
        
        # Category distribution
        cat_dist = llm.get('category_distribution', {})
        if cat_dist:
            lines.extend([
                "### Primary Category Distribution",
                "",
                "| Category | Count | % |",
                "|----------|-------|---|",
            ])
            for cat, stats in cat_dist.items():
                cat_name = CATEGORY_NAMES.get(cat, cat)
                lines.append(f"| {cat_name} | {stats.get('count', 0)} | {stats.get('pct', 0)}% |")
            lines.append("")
        
        # Primary code distribution
        code_dist = llm.get('primary_code_distribution', {})
        if code_dist:
            lines.extend([
                "### Primary Failure Codes",
                "",
                "| Code | Count | % | Description |",
                "|------|-------|---|-------------|",
            ])
            for code, stats in code_dist.items():
                desc = CODE_DESCRIPTIONS.get(code, code)
                lines.append(f"| {code} | {stats.get('count', 0)} | {stats.get('pct', 0)}% | {desc} |")
            lines.append("")
        
        # All detected codes (including secondary)
        all_codes = llm.get('all_codes_distribution', {})
        if all_codes:
            lines.extend([
                "### All Detected Codes (Primary + Secondary)",
                "",
                "| Code | Count | % | Description |",
                "|------|-------|---|-------------|",
            ])
            for code, stats in list(all_codes.items())[:15]:
                desc = CODE_DESCRIPTIONS.get(code, code)
                lines.append(f"| {code} | {stats.get('count', 0)} | {stats.get('pct', 0)}% | {desc} |")
            lines.append("")
        
        # Direction distribution
        dir_dist = llm.get('direction_distribution', {})
        if dir_dist:
            lines.extend([
                "### Direction Distribution",
                "",
                "| Direction | Count | % |",
                "|-----------|-------|---|",
            ])
            for direction, stats in dir_dist.items():
                lines.append(f"| {direction} | {stats.get('count', 0)} | {stats.get('pct', 0)}% |")
            lines.append("")
        
        # By flip type (deference vs skepticism)
        by_flip = llm.get('by_flip_type', {})
        if by_flip:
            lines.extend([
                "### Deference vs Skepticism Analysis",
                "",
            ])
            for ft, ft_data in by_flip.items():
                top_code = ft_data.get('top_primary_code')
                top_cat = ft_data.get('top_category')
                lines.extend([
                    f"#### {ft.upper()} (n={ft_data.get('valid', 0)})",
                    "",
                    f"- **Top Code**: {top_code[0] if top_code else 'N/A'} ({top_code[1] if top_code else 0})",
                    f"- **Top Category**: {top_cat[0] if top_cat else 'N/A'} ({top_cat[1] if top_cat else 0})",
                    "",
                ])
                
                # Top codes for this flip type
                codes = ft_data.get('primary_codes', {})
                if codes:
                    lines.extend([
                        "| Code | Count | % |",
                        "|------|-------|---|",
                    ])
                    for code, code_stats in list(codes.items())[:5]:
                        lines.append(f"| {code} | {code_stats.get('count', 0)} | {code_stats.get('pct', 0)}% |")
                    lines.append("")
        
        # By dataset
        by_ds = llm.get('by_dataset', {})
        if by_ds:
            lines.extend([
                "### By Dataset",
                "",
                "| Dataset | Flips | Top Category | Top Code |",
                "|---------|-------|--------------|----------|",
            ])
            for ds, ds_data in sorted(by_ds.items()):
                top_code = ds_data.get('top_primary_code')
                top_cat = ds_data.get('top_category')
                lines.append(
                    f"| {ds} | {ds_data.get('total_flips', 0)} | "
                    f"{top_cat[0] if top_cat else 'N/A'} | {top_code[0] if top_code else 'N/A'} |"
                )
            lines.append("")
        
        # By model
        by_model = llm.get('by_model', {})
        if by_model and len(by_model) > 1:
            lines.extend([
                "### By Model",
                "",
                "| Model | Valid | Errors | Top Category | Top Code |",
                "|-------|-------|--------|--------------|----------|",
            ])
            for model, m_data in by_model.items():
                top_code = m_data.get('top_primary_code')
                top_cat = m_data.get('top_category')
                lines.append(
                    f"| {model} | {m_data.get('valid', 0)} | {m_data.get('errors', 0)} | "
                    f"{top_cat[0] if top_cat else 'N/A'} | {top_code[0] if top_code else 'N/A'} |"
                )
            lines.append("")
        
        # Code co-occurrence
        cooccur = llm.get('code_cooccurrence', {})
        if cooccur:
            lines.extend([
                "### Code Co-occurrence (Top 5 per code)",
                "",
            ])
            for code, co_codes in list(cooccur.items())[:8]:
                co_str = ", ".join([f"{c}({n})" for c, n in list(co_codes.items())[:3]])
                lines.append(f"- **{code}**: {co_str}")
            lines.append("")
    
    # Footer
    lines.extend([
        "---",
        "",
        "*Report generated by DialDefer Taxonomy Pipeline*",
    ])
    
    return "\n".join(lines)


def generate_latex_tables(data: Dict) -> str:
    """Generate LaTeX tables for academic papers."""
    
    lines = [
        "% DialDefer Taxonomy Analysis - LaTeX Tables",
        "% Auto-generated",
        "",
        "% Required packages:",
        "% \\usepackage{booktabs}",
        "% \\usepackage{multirow}",
        "",
    ]
    
    # Extraction summary table
    ext = data.get('extraction_stats', {})
    meta = data.get('metadata', {})
    
    lines.extend([
        "% ============================================================",
        "% TABLE 1: Flip Extraction Summary",
        "% ============================================================",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Deference & Skepticism & Total \\\\",
        "\\midrule",
    ])
    
    by_model = ext.get('by_model', {})
    total_def, total_skep = 0, 0
    for model, stats in by_model.items():
        model_disp = model.replace("_", "\\_")
        d = stats.get('deference', 0)
        s = stats.get('skepticism', 0)
        total_def += d
        total_skep += s
        lines.append(f"{model_disp} & {d} & {s} & {d + s} \\\\")
    
    lines.extend([
        "\\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total_def}}} & \\textbf{{{total_skep}}} & \\textbf{{{total_def + total_skep}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Judgment flip extraction summary across {len(by_model)} model(s).}}",
        "\\label{tab:flip_extraction}",
        "\\end{table}",
        "",
    ])
    
    # Quantitative analysis table
    quant = data.get('quantitative_analysis', {})
    per_model = quant.get('per_model', {})
    
    if per_model:
        lines.extend([
            "% ============================================================",
            "% TABLE 2: Quantitative Analysis Summary",
            "% ============================================================",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Model & N & Flips & Flip Rate & DDS$\\mu$ & DDS$\\sigma$ \\\\",
            "\\midrule",
        ])
        
        for model, stats in per_model.items():
            model_disp = model.replace("_", "\\_")
            lines.append(
                f"{model_disp} & {stats.get('n_items', 0)} & "
                f"{stats.get('n_any_flip', 0)} & {stats.get('flip_rate', 0):.1f}\\% & "
                f"{stats.get('dds_mean', 0):.3f} & {stats.get('dds_std', 0):.3f} \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Quantitative analysis: flip rates and Dialogic Deference Score (DDS).}",
            "\\label{tab:quantitative_summary}",
            "\\end{table}",
            "",
        ])
    
    # LLM Taxonomy tables
    llm = data.get('llm_taxonomy', {})
    
    if llm:
        # Category distribution
        cat_dist = llm.get('category_distribution', {})
        if cat_dist:
            llm_meta = llm.get('metadata', {})
            n_valid = llm_meta.get('valid_analyses', 0)
            
            lines.extend([
                "% ============================================================",
                "% TABLE 3: Failure Category Distribution",
                "% ============================================================",
                "",
                "\\begin{table}[t]",
                "\\centering",
                "\\begin{tabular}{lrr}",
                "\\toprule",
                "Category & Count & \\% \\\\",
                "\\midrule",
            ])
            
            for cat, stats in cat_dist.items():
                cat_disp = CATEGORY_NAMES.get(cat, cat).replace("_", " ")
                lines.append(f"{cat_disp} & {stats.get('count', 0)} & {stats.get('pct', 0)}\\% \\\\")
            
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                f"\\caption{{Primary failure category distribution (N={n_valid}).}}",
                "\\label{tab:category_distribution}",
                "\\end{table}",
                "",
            ])
        
        # Primary code distribution
        code_dist = llm.get('primary_code_distribution', {})
        if code_dist:
            lines.extend([
                "% ============================================================",
                "% TABLE 4: Primary Failure Codes",
                "% ============================================================",
                "",
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                "\\begin{tabular}{clrr}",
                "\\toprule",
                "Code & Failure Type & Count & \\% \\\\",
                "\\midrule",
            ])
            
            for code, stats in list(code_dist.items())[:12]:
                desc = CODE_DESCRIPTIONS.get(code, code).split(" - ")[0] if " - " in CODE_DESCRIPTIONS.get(code, code) else CODE_DESCRIPTIONS.get(code, code)[:30]
                desc = desc.replace("_", "\\_")
                lines.append(f"{code} & {desc} & {stats.get('count', 0)} & {stats.get('pct', 0)}\\% \\\\")
            
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Top primary failure codes from LLM-as-judge analysis.}",
                "\\label{tab:failure_codes}",
                "\\end{table}",
                "",
            ])
        
        # Deference vs Skepticism comparison
        by_flip = llm.get('by_flip_type', {})
        if by_flip and len(by_flip) == 2:
            lines.extend([
                "% ============================================================",
                "% TABLE 5: Deference vs Skepticism",
                "% ============================================================",
                "",
                "\\begin{table}[t]",
                "\\centering",
                "\\begin{tabular}{llrr}",
                "\\toprule",
                " & & Deference & Skepticism \\\\",
                "\\midrule",
            ])
            
            def_data = by_flip.get('deference', {})
            skep_data = by_flip.get('skepticism', {})
            
            lines.append(f"\\multicolumn{{2}}{{l}}{{Total Flips}} & {def_data.get('valid', 0)} & {skep_data.get('valid', 0)} \\\\")
            
            def_top = def_data.get('top_primary_code', ('N/A', 0))
            skep_top = skep_data.get('top_primary_code', ('N/A', 0))
            lines.append(f"\\multicolumn{{2}}{{l}}{{Top Code}} & {def_top[0]} ({def_top[1]}) & {skep_top[0]} ({skep_top[1]}) \\\\")
            
            def_cat = def_data.get('top_category', ('N/A', 0))
            skep_cat = skep_data.get('top_category', ('N/A', 0))
            cat1 = def_cat[0].replace("_", " ")[:15] if def_cat[0] else 'N/A'
            cat2 = skep_cat[0].replace("_", " ")[:15] if skep_cat[0] else 'N/A'
            lines.append(f"\\multicolumn{{2}}{{l}}{{Top Category}} & {cat1} & {cat2} \\\\")
            
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Comparison of failure patterns between deference and skepticism flips.}",
                "\\label{tab:deference_vs_skepticism}",
                "\\end{table}",
                "",
            ])
    
    # Cross-model table (if available)
    cross = data.get('cross_model_analysis', {})
    if cross and cross.get('total_common_items', 0) > 0:
        summary = cross.get('summary', {})
        models = cross.get('models_analyzed', [])
        
        lines.extend([
            "% ============================================================",
            "% TABLE 6: Cross-Model Agreement",
            "% ============================================================",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lr}",
            "\\toprule",
            "Metric & Value \\\\",
            "\\midrule",
            f"Models Compared & {len(models)} \\\\",
            f"Common Items & {cross.get('total_common_items', 0)} \\\\",
            f"Flip in ALL models & {summary.get('items_flip_all_models', 0)} ({summary.get('universal_flip_rate', 0)}\\%) \\\\",
            f"Flip in NO models & {summary.get('items_flip_no_models', 0)} \\\\",
            f"Agreement Rate & {summary.get('agreement_rate', 0)}\\% \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Cross-model flip agreement ({', '.join(models)}).}}",
            "\\label{tab:cross_model}",
            "\\end{table}",
            "",
        ])
    
    return "\n".join(lines)


def generate_csv_summary(data: Dict) -> str:
    """Generate CSV summary for easy analysis."""
    
    lines = ["metric,model,dataset,value"]
    
    # Quantitative metrics
    quant = data.get('quantitative_analysis', {})
    domain_stats = quant.get('domain_stats_by_model', {})
    
    for model, ds_stats in domain_stats.items():
        for ds, s in ds_stats.items():
            lines.append(f"dds,{model},{ds},{s.get('dds', 0):.3f}")
            lines.append(f"flip_rate,{model},{ds},{s.get('flip_rate', 0):.3f}")
            lines.append(f"c1_acc,{model},{ds},{s.get('c1_avg_acc', 0):.3f}")
            lines.append(f"c2_acc,{model},{ds},{s.get('c2_avg_acc', 0):.3f}")
    
    # LLM taxonomy codes
    llm = data.get('llm_taxonomy', {})
    if llm:
        by_model = llm.get('by_model', {})
        for model, m_data in by_model.items():
            for code, code_stats in m_data.get('primary_codes', {}).items():
                lines.append(f"code_{code},{model},all,{code_stats.get('count', 0)}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports from taxonomy JSON output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all reports (markdown + latex + csv)
    python generate_reports.py taxonomy_output.json
    
    # Custom output name
    python generate_reports.py taxonomy_output.json -o my_report
    
    # Only LaTeX
    python generate_reports.py taxonomy_output.json --latex-only
    
    # Only Markdown
    python generate_reports.py taxonomy_output.json --markdown-only
        """
    )
    
    parser.add_argument('input_json', help='Input JSON file from run_full_taxonomy.py')
    parser.add_argument('-o', '--output', help='Output base name (default: input name)')
    parser.add_argument('--latex-only', action='store_true', help='Generate only LaTeX')
    parser.add_argument('--markdown-only', action='store_true', help='Generate only Markdown')
    parser.add_argument('--csv', action='store_true', help='Also generate CSV summary')
    
    args = parser.parse_args()
    
    # Load JSON
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return
    
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Determine output base name
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = input_path.with_suffix('')
    
    # Generate reports
    generated = []
    
    if not args.latex_only:
        # Markdown report
        md_path = output_base.with_suffix('.md')
        md_content = generate_markdown_report(data)
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"Markdown report: {md_path}")
        generated.append(md_path)
    
    if not args.markdown_only:
        # LaTeX tables
        tex_path = output_base.with_name(output_base.stem + '_tables.tex')
        tex_content = generate_latex_tables(data)
        with open(tex_path, 'w') as f:
            f.write(tex_content)
        print(f"LaTeX tables: {tex_path}")
        generated.append(tex_path)
    
    if args.csv:
        # CSV summary
        csv_path = output_base.with_suffix('.csv')
        csv_content = generate_csv_summary(data)
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        print(f"CSV summary: {csv_path}")
        generated.append(csv_path)
    
    print(f"\nGenerated {len(generated)} report(s)")


if __name__ == "__main__":
    main()

