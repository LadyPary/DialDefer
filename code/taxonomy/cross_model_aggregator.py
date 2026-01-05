#!/usr/bin/env python3
"""
DialDefer Cross-Model Aggregation Analysis
==========================================

Generates all cross-model comparison tables, statistics, and LaTeX output.

Usage:
    python cross_model_aggregator.py \
        --models gpt4o_analysis.json nova_analysis.json qwen_analysis.json \
        --names "GPT-4o" "Nova-Lite" "Qwen-2.5-7B" \
        --output results/

    # Or use CSV files directly:
    python cross_model_aggregator.py \
        --summaries gpt4o_model_analysis_summary.csv nova-lite-1.0_model_analysis_summary.csv qwen-2.5-7b-instruct_model_analysis_summary.csv \
        --items gpt4o_model_analysis_items.csv nova-lite-1.0_model_analysis_items.csv qwen-2.5-7b-instruct_model_analysis_items.csv \
        --names "GPT-4o" "Nova-Lite" "Qwen-2.5-7B" \
        --output results/
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CrossModelAnalyzer:
    """Analyze and compare results across multiple models."""
    
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.summaries = {}  # model_name -> DataFrame
        self.items = {}      # model_name -> DataFrame
        self.results = {}    # Store computed results
        
    def load_from_json(self, json_files: List[str]):
        """Load from JSON analysis files."""
        for name, filepath in zip(self.model_names, json_files):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrames
            self.summaries[name] = pd.DataFrame(data['domain_stats']).T.reset_index()
            self.summaries[name].rename(columns={'index': 'dataset'}, inplace=True)
            self.items[name] = pd.DataFrame(data['items'])
            
    def load_from_csv(self, summary_files: List[str], item_files: List[str]):
        """Load from CSV files."""
        for name, summ_file, item_file in zip(self.model_names, summary_files, item_files):
            self.summaries[name] = pd.read_csv(summ_file)
            self.items[name] = pd.read_csv(item_file)
    
    def _create_item_keys(self):
        """Create unique keys for item matching."""
        for name in self.model_names:
            self.items[name]['key'] = (
                self.items[name]['dataset'].astype(str) + '::' + 
                self.items[name]['id'].astype(str)
            )
    
    def compute_all(self) -> Dict:
        """Compute all cross-model statistics."""
        self._create_item_keys()
        
        self.results = {
            'dds_comparison': self._compute_dds_comparison(),
            'variance_comparison': self._compute_variance_comparison(),
            'flip_comparison': self._compute_flip_comparison(),
            'item_agreement': self._compute_item_agreement(),
            'item_correlations': self._compute_item_correlations(),
            'model_summary': self._compute_model_summary(),
            'consistent_domains': self._identify_consistent_domains(),
        }
        
        return self.results
    
    def _compute_dds_comparison(self) -> pd.DataFrame:
        """Compare DDS across models by dataset."""
        datasets = sorted(set(self.summaries[self.model_names[0]]['dataset']))
        
        rows = []
        for ds in datasets:
            row = {'dataset': ds}
            values = []
            for name in self.model_names:
                dds = self.summaries[name][self.summaries[name]['dataset'] == ds]['dds'].values[0]
                row[f'{name}_dds'] = dds
                values.append(dds)
            
            row['mean_dds'] = np.mean(values)
            row['std_dds'] = np.std(values)
            row['all_positive'] = all(v > 0 for v in values)
            row['all_negative'] = all(v < 0 for v in values)
            row['consistent'] = row['all_positive'] or row['all_negative']
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _compute_variance_comparison(self) -> pd.DataFrame:
        """Compare within-domain variance across models."""
        datasets = sorted(set(self.summaries[self.model_names[0]]['dataset']))
        
        rows = []
        for ds in datasets:
            row = {'dataset': ds}
            values = []
            for name in self.model_names:
                var = self.summaries[name][self.summaries[name]['dataset'] == ds]['dds_var'].values[0]
                row[f'{name}_var'] = var
                values.append(var)
            
            row['mean_var'] = np.mean(values)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _compute_flip_comparison(self) -> pd.DataFrame:
        """Compare flip rates across models."""
        datasets = sorted(set(self.summaries[self.model_names[0]]['dataset']))
        
        rows = []
        for ds in datasets:
            row = {'dataset': ds}
            for name in self.model_names:
                summ = self.summaries[name][self.summaries[name]['dataset'] == ds]
                row[f'{name}_flip_rate'] = summ['flip_rate'].values[0]
                row[f'{name}_def_flips'] = summ['n_deference_flips'].values[0]
                row[f'{name}_skep_flips'] = summ['n_skepticism_flips'].values[0]
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _compute_item_agreement(self) -> Dict:
        """Compute cross-model item-level agreement."""
        # Find common items
        common_keys = set(self.items[self.model_names[0]]['key'])
        for name in self.model_names[1:]:
            common_keys &= set(self.items[name]['key'])
        
        # Analyze agreement
        by_dataset = defaultdict(lambda: {'all_flip': 0, 'none_flip': 0, 'mixed': 0, 'total': 0})
        consistent_flips = []
        
        for key in common_keys:
            flip_statuses = []
            dataset = None
            for name in self.model_names:
                row = self.items[name][self.items[name]['key'] == key].iloc[0]
                flip_statuses.append(row['any_flip'])
                dataset = row['dataset']
            
            by_dataset[dataset]['total'] += 1
            
            if all(flip_statuses):
                by_dataset[dataset]['all_flip'] += 1
                consistent_flips.append({
                    'key': key,
                    'dataset': dataset,
                })
            elif not any(flip_statuses):
                by_dataset[dataset]['none_flip'] += 1
            else:
                by_dataset[dataset]['mixed'] += 1
        
        # Totals
        total_all = sum(d['all_flip'] for d in by_dataset.values())
        total_none = sum(d['none_flip'] for d in by_dataset.values())
        total_mixed = sum(d['mixed'] for d in by_dataset.values())
        total = len(common_keys)
        
        return {
            'total_items': total,
            'all_flip': total_all,
            'none_flip': total_none,
            'mixed': total_mixed,
            'agreement_rate': (total_all + total_none) / total * 100,
            'by_dataset': dict(by_dataset),
            'consistent_flips': consistent_flips,
        }
    
    def _compute_item_correlations(self) -> Dict:
        """Compute item-level DDS correlations between models."""
        # Merge all items on key
        merged = self.items[self.model_names[0]][['key', 'item_dds']].copy()
        merged.columns = ['key', f'{self.model_names[0]}_dds']
        
        for name in self.model_names[1:]:
            temp = self.items[name][['key', 'item_dds']].copy()
            temp.columns = ['key', f'{name}_dds']
            merged = merged.merge(temp, on='key', how='inner')
        
        # Compute correlations
        correlations = {}
        for i, name1 in enumerate(self.model_names):
            for name2 in self.model_names[i+1:]:
                col1 = f'{name1}_dds'
                col2 = f'{name2}_dds'
                r = merged[col1].corr(merged[col2])
                correlations[f'{name1}_vs_{name2}'] = r
        
        return {
            'correlations': correlations,
            'n_items': len(merged),
        }
    
    def _compute_model_summary(self) -> pd.DataFrame:
        """Compute overall summary statistics per model."""
        rows = []
        for name in self.model_names:
            summ = self.summaries[name]
            items = self.items[name]
            
            # Weighted DDS by n_items
            weighted_dds = (summ['dds'] * summ['n_items']).sum() / summ['n_items'].sum()
            
            rows.append({
                'model': name,
                'n_items': summ['n_items'].sum(),
                'n_datasets': len(summ),
                'mean_dds': weighted_dds,
                'mean_flip_rate': summ['flip_rate'].mean(),
                'mean_variance': summ['dds_var'].mean(),
                'total_def_flips': summ['n_deference_flips'].sum(),
                'total_skep_flips': summ['n_skepticism_flips'].sum(),
            })
        
        return pd.DataFrame(rows)
    
    def _identify_consistent_domains(self) -> Dict:
        """Identify domains with consistent behavior across models."""
        dds_comp = self._compute_dds_comparison()
        
        return {
            'all_deference': dds_comp[dds_comp['all_positive']]['dataset'].tolist(),
            'all_skepticism': dds_comp[dds_comp['all_negative']]['dataset'].tolist(),
            'inconsistent': dds_comp[~dds_comp['consistent']]['dataset'].tolist(),
        }
    
    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================
    
    def to_latex_tables(self) -> str:
        """Generate LaTeX tables for paper."""
        latex = []
        
        # Table 1: DDS Comparison
        latex.append(self._latex_dds_table())
        
        # Table 2: Variance Comparison
        latex.append(self._latex_variance_table())
        
        # Table 3: Item Agreement
        latex.append(self._latex_agreement_table())
        
        # Table 4: Model Summary
        latex.append(self._latex_summary_table())
        
        return '\n\n'.join(latex)
    
    def _latex_dds_table(self) -> str:
        """Generate DDS comparison table."""
        dds = self.results['dds_comparison']
        
        # Build header
        model_cols = ' & '.join(self.model_names)
        header = f"Dataset & {model_cols} & Mean & Consistent"
        
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            f"\\begin{{tabular}}{{l|{'c' * len(self.model_names)}|cc}}",
            "\\toprule",
            f"{header} \\\\",
            "\\midrule",
        ]
        
        for _, row in dds.iterrows():
            values = [f"{row[f'{name}_dds']:+.1f}" for name in self.model_names]
            consistent = "\\checkmark" if row['consistent'] else ""
            line = f"{row['dataset']} & {' & '.join(values)} & {row['mean_dds']:+.1f} & {consistent} \\\\"
            lines.append(line)
        
        # Average row
        avg_values = [f"{dds[f'{name}_dds'].mean():+.1f}" for name in self.model_names]
        lines.append("\\midrule")
        lines.append(f"\\textbf{{Average}} & {' & '.join(avg_values)} & {dds['mean_dds'].mean():+.1f} & \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Cross-model DDS comparison. Positive values indicate dialogic deference; negative values indicate skepticism. Checkmark indicates consistent direction across all models.}",
            "\\label{tab:cross_model_dds}",
            "\\end{table}",
        ])
        
        return '\n'.join(lines)
    
    def _latex_variance_table(self) -> str:
        """Generate variance comparison table."""
        var = self.results['variance_comparison']
        
        model_cols = ' & '.join(self.model_names)
        header = f"Dataset & {model_cols} & Mean"
        
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            f"\\begin{{tabular}}{{l|{'c' * len(self.model_names)}|c}}",
            "\\toprule",
            f"{header} \\\\",
            "\\midrule",
        ]
        
        for _, row in var.iterrows():
            values = [f"{row[f'{name}_var']:.3f}" for name in self.model_names]
            line = f"{row['dataset']} & {' & '.join(values)} & {row['mean_var']:.3f} \\\\"
            lines.append(line)
        
        # Average row
        avg_values = [f"{var[f'{name}_var'].mean():.3f}" for name in self.model_names]
        lines.append("\\midrule")
        lines.append(f"\\textbf{{Average}} & {' & '.join(avg_values)} & {var['mean_var'].mean():.3f} \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Within-domain DDS variance across models. Lower values indicate more consistent behavior within each domain.}",
            "\\label{tab:cross_model_variance}",
            "\\end{table}",
        ])
        
        return '\n'.join(lines)
    
    def _latex_agreement_table(self) -> str:
        """Generate item agreement table."""
        agree = self.results['item_agreement']
        
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Agreement Type & Count & Percentage \\\\",
            "\\midrule",
            f"All models flip & {agree['all_flip']} & {agree['all_flip']/agree['total_items']*100:.1f}\\% \\\\",
            f"No model flips & {agree['none_flip']} & {agree['none_flip']/agree['total_items']*100:.1f}\\% \\\\",
            f"Mixed (1--2 flip) & {agree['mixed']} & {agree['mixed']/agree['total_items']*100:.1f}\\% \\\\",
            "\\midrule",
            f"\\textbf{{Total}} & {agree['total_items']} & 100\\% \\\\",
            f"\\textbf{{Agreement Rate}} & & \\textbf{{{agree['agreement_rate']:.1f}\\%}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Cross-model item agreement (N={agree['total_items']}). Agreement rate measures items where all models agree on flip/no-flip.}}",
            "\\label{tab:item_agreement}",
            "\\end{table}",
        ]
        
        return '\n'.join(lines)
    
    def _latex_summary_table(self) -> str:
        """Generate model summary table."""
        summ = self.results['model_summary']
        
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Model & Items & Mean DDS & Flip Rate & Variance & Def/Skep \\\\",
            "\\midrule",
        ]
        
        for _, row in summ.iterrows():
            def_skep = f"{int(row['total_def_flips'])}/{int(row['total_skep_flips'])}"
            line = f"{row['model']} & {int(row['n_items'])} & {row['mean_dds']:+.1f} & {row['mean_flip_rate']:.1f}\\% & {row['mean_variance']:.3f} & {def_skep} \\\\"
            lines.append(line)
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Overall model comparison. Def/Skep shows total deference vs. skepticism flips.}",
            "\\label{tab:model_summary}",
            "\\end{table}",
        ])
        
        return '\n'.join(lines)
    
    def to_markdown_report(self) -> str:
        """Generate full markdown report."""
        report = []
        
        # Header
        report.append("# Cross-Model Quantitative Analysis Report\n")
        
        # Executive Summary
        summ = self.results['model_summary']
        report.append("## Executive Summary\n")
        report.append(f"Analysis of {len(self.model_names)} models across {summ['n_datasets'].iloc[0]} domains:\n")
        for _, row in summ.iterrows():
            report.append(f"- **{row['model']}**: Mean DDS = {row['mean_dds']:+.1f}, Flip Rate = {row['mean_flip_rate']:.1f}%")
        report.append("")
        
        # Key Finding
        agree = self.results['item_agreement']
        report.append(f"**Key Finding**: Only {agree['all_flip']}/{agree['total_items']} items ({agree['all_flip']/agree['total_items']*100:.1f}%) flip in ALL models.")
        report.append("Dialogic deference is largely **model-specific**, not item-inherent.\n")
        
        # DDS Comparison
        report.append("## 1. Cross-Model DDS Comparison\n")
        dds = self.results['dds_comparison']
        
        header = "| Dataset | " + " | ".join(self.model_names) + " | Mean | Consistent |"
        sep = "|" + "|".join(["---"] * (len(self.model_names) + 3)) + "|"
        report.append(header)
        report.append(sep)
        
        for _, row in dds.iterrows():
            values = [f"{row[f'{name}_dds']:+.1f}" for name in self.model_names]
            consistent = "[ok]" if row['consistent'] else "[x]"
            report.append(f"| {row['dataset']} | " + " | ".join(values) + f" | {row['mean_dds']:+.1f} | {consistent} |")
        
        report.append("")
        
        # Consistent domains
        cons = self.results['consistent_domains']
        report.append(f"**Consistent deference domains**: {', '.join(cons['all_deference'])}\n")
        report.append(f"**Inconsistent domains**: {', '.join(cons['inconsistent'])}\n")
        
        # Variance
        report.append("## 2. Within-Domain Variance\n")
        var = self.results['variance_comparison']
        
        header = "| Dataset | " + " | ".join(self.model_names) + " | Mean |"
        sep = "|" + "|".join(["---"] * (len(self.model_names) + 2)) + "|"
        report.append(header)
        report.append(sep)
        
        for _, row in var.iterrows():
            values = [f"{row[f'{name}_var']:.3f}" for name in self.model_names]
            report.append(f"| {row['dataset']} | " + " | ".join(values) + f" | {row['mean_var']:.3f} |")
        
        report.append("")
        
        # Item Agreement
        report.append("## 3. Cross-Model Item Agreement\n")
        report.append("| Metric | Count | Percentage |")
        report.append("|--------|-------|------------|")
        report.append(f"| All models flip | {agree['all_flip']} | {agree['all_flip']/agree['total_items']*100:.1f}% |")
        report.append(f"| No model flips | {agree['none_flip']} | {agree['none_flip']/agree['total_items']*100:.1f}% |")
        report.append(f"| Mixed (1-2 flip) | {agree['mixed']} | {agree['mixed']/agree['total_items']*100:.1f}% |")
        report.append(f"| **Agreement Rate** | | **{agree['agreement_rate']:.1f}%** |")
        report.append("")
        
        # Correlations
        report.append("## 4. Item-Level DDS Correlations\n")
        corr = self.results['item_correlations']
        report.append("| Model Pair | Correlation (r) |")
        report.append("|------------|-----------------|")
        for pair, r in corr['correlations'].items():
            report.append(f"| {pair.replace('_', ' ')} | {r:.3f} |")
        report.append("")
        report.append("Low correlations indicate model-specific rather than item-specific behavior.\n")
        
        # Key Findings
        report.append("## 5. Key Findings\n")
        report.append("1. **Systematic but Model-Dependent**: Low variance within domains, low correlation across models")
        report.append(f"2. **{len(cons['all_deference'])}/{len(dds)} domains** show consistent deference across all models")
        report.append(f"3. **Only {agree['all_flip']} items** are universally problematic (flip in all models)")
        report.append(f"4. **{agree['agreement_rate']:.1f}% agreement rate** on flip/no-flip decisions")
        
        return '\n'.join(report)
    
    def save_all(self, output_dir: str):
        """Save all outputs to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSVs
        self.results['dds_comparison'].to_csv(output_dir / 'cross_model_dds.csv', index=False)
        self.results['variance_comparison'].to_csv(output_dir / 'cross_model_variance.csv', index=False)
        self.results['flip_comparison'].to_csv(output_dir / 'cross_model_flips.csv', index=False)
        self.results['model_summary'].to_csv(output_dir / 'model_summary.csv', index=False)
        
        # Save JSON
        # Convert non-serializable items
        json_results = {
            'model_names': self.model_names,
            'item_agreement': {
                k: v for k, v in self.results['item_agreement'].items() 
                if k != 'consistent_flips'
            },
            'item_correlations': self.results['item_correlations'],
            'consistent_domains': self.results['consistent_domains'],
        }
        with open(output_dir / 'cross_model_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save LaTeX
        with open(output_dir / 'latex_tables.tex', 'w') as f:
            f.write(self.to_latex_tables())
        
        # Save Markdown report
        with open(output_dir / 'CROSS_MODEL_REPORT.md', 'w') as f:
            f.write(self.to_markdown_report())
        
        print(f"Saved all outputs to {output_dir}/")
        print(f"  - cross_model_dds.csv")
        print(f"  - cross_model_variance.csv")
        print(f"  - cross_model_flips.csv")
        print(f"  - model_summary.csv")
        print(f"  - cross_model_results.json")
        print(f"  - latex_tables.tex")
        print(f"  - CROSS_MODEL_REPORT.md")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model aggregation analysis for DialDefer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using JSON files
    python cross_model_aggregator.py \\
        --models gpt4o.json nova.json qwen.json \\
        --names "GPT-4o" "Nova-Lite" "Qwen-2.5" \\
        --output results/
    
    # Using CSV files
    python cross_model_aggregator.py \\
        --summaries gpt4o_summary.csv nova_summary.csv qwen_summary.csv \\
        --items gpt4o_items.csv nova_items.csv qwen_items.csv \\
        --names "GPT-4o" "Nova-Lite" "Qwen-2.5" \\
        --output results/
        """
    )
    
    parser.add_argument('--models', '-m', nargs='+', help='JSON analysis files')
    parser.add_argument('--summaries', '-s', nargs='+', help='Summary CSV files')
    parser.add_argument('--items', '-i', nargs='+', help='Items CSV files')
    parser.add_argument('--names', '-n', nargs='+', required=True, help='Model names')
    parser.add_argument('--output', '-o', default='cross_model_results', help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = CrossModelAnalyzer(args.names)
    
    if args.models:
        analyzer.load_from_json(args.models)
    elif args.summaries and args.items:
        analyzer.load_from_csv(args.summaries, args.items)
    else:
        parser.error("Provide either --models (JSON) or both --summaries and --items (CSV)")
    
    # Compute all statistics
    analyzer.compute_all()
    
    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL ANALYSIS COMPLETE")
    print("=" * 70)
    
    summ = analyzer.results['model_summary']
    print("\nModel Summary:")
    print(summ.to_string(index=False))
    
    agree = analyzer.results['item_agreement']
    print(f"\nItem Agreement: {agree['agreement_rate']:.1f}%")
    print(f"  All flip: {agree['all_flip']} | None flip: {agree['none_flip']} | Mixed: {agree['mixed']}")
    
    # Save outputs
    analyzer.save_all(args.output)


if __name__ == "__main__":
    main()
