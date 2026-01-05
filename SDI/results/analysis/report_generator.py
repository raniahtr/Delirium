"""
Generate markdown analysis reports summarizing SDI analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to markdown table format without requiring tabulate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to convert
    
    Returns:
    --------
    markdown : str
        Markdown-formatted table
    """
    if len(df) == 0:
        return "*(No data)*"
    
    # Get column names
    cols = df.columns.tolist()
    
    # Format header
    header = "| " + " | ".join(str(col) for col in cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    
    # Format rows
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[col]) if pd.notna(row[col]) else "N/A" for col in cols) + " |"
        rows.append(row_str)
    
    return "\n".join([header, separator] + rows)


def generate_analysis_report(
    results: Dict,
    output_path: str,
    band: str = "bandpass"
):
    """
    Generate comprehensive markdown analysis report.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_analysis()
    output_path : str
        Path to save report
    band : str
        Frequency band
    """
    report_lines = []
    
    # Header
    report_lines.append(f"# SDI Analysis Report: {band}")
    report_lines.append("")
    report_lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Data Overview
    report_lines.append("## Data Overview")
    report_lines.append("")
    
    unified_df = results.get('unified_df')
    if unified_df is not None:
        groups = sorted(unified_df['group'].unique())
        report_lines.append(f"**Groups analyzed:** {', '.join(groups)}")
        report_lines.append("")
        report_lines.append("| Group | N Subjects | N Parcels | N Observations |")
        report_lines.append("|-------|------------|-----------|----------------|")
        
        for group in groups:
            group_df = unified_df[unified_df['group'] == group]
            n_subjects = group_df['subject_id'].nunique()
            n_parcels = group_df['parcel_id'].nunique()
            n_obs = len(group_df)
            report_lines.append(f"| {group} | {n_subjects} | {n_parcels} | {n_obs} |")
        
        report_lines.append("")
    
    # Assumption Checks
    report_lines.append("## Statistical Assumption Checks")
    report_lines.append("")
    
    assumption_results = results.get('assumption_checks', {})
    if assumption_results:
        recommendations = assumption_results.get('recommendations', {})
        report_lines.append(f"**Recommended test type:** {recommendations.get('primary_test', 'N/A')}")
        report_lines.append(f"**All groups normal:** {recommendations.get('all_normal', 'N/A')}")
        report_lines.append(f"**Variances homogeneous:** {recommendations.get('variances_homogeneous', 'N/A')}")
        report_lines.append("")
        
        # Normality tests
        if assumption_results.get('normality_tests'):
            report_lines.append("### Normality Tests")
            report_lines.append("")
            norm_df = pd.DataFrame(assumption_results['normality_tests'])
            # Use simple markdown table format (doesn't require tabulate)
            report_lines.append(_dataframe_to_markdown(norm_df))
            report_lines.append("")
        
        # Variance tests
        if assumption_results.get('variance_tests'):
            report_lines.append("### Variance Tests")
            report_lines.append("")
            for test_name, test_result in assumption_results['variance_tests'].items():
                if test_result:
                    report_lines.append(f"**{test_name.capitalize()} test:**")
                    report_lines.append(f"- Statistic: {test_result.get('statistic', 'N/A')}")
                    report_lines.append(f"- p-value: {test_result.get('pvalue', 'N/A')}")
                    report_lines.append(f"- Homogeneous: {test_result.get('is_homogeneous', 'N/A')}")
                    report_lines.append("")
    
    # Group Comparisons
    report_lines.append("## Group-Level Comparisons")
    report_lines.append("")
    
    group_comparisons = results.get('group_comparisons', {})
    if group_comparisons:
        primary_test = group_comparisons.get('primary_test', {})
        if primary_test:
            report_lines.append("### Primary Test")
            report_lines.append("")
            report_lines.append(f"**Test:** {primary_test.get('test_name', 'N/A')}")
            report_lines.append(f"**Statistic:** {primary_test.get('statistic', 'N/A')}")
            report_lines.append(f"**p-value:** {primary_test.get('pvalue', 'N/A')}")
            report_lines.append("")
        
        posthoc_df = group_comparisons.get('posthoc_tests', pd.DataFrame())
        if len(posthoc_df) > 0:
            report_lines.append("### Post-hoc Tests")
            report_lines.append("")
            # Create summary table
            summary_cols = ['group1', 'group2', 'pvalue_corrected', 'effect_size', 'significant']
            if all(col in posthoc_df.columns for col in summary_cols):
                summary_df = posthoc_df[summary_cols].copy()
                summary_df['pvalue_corrected'] = summary_df['pvalue_corrected'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                summary_df['effect_size'] = summary_df['effect_size'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                report_lines.append(_dataframe_to_markdown(summary_df))
            report_lines.append("")
    
    # Parcel-Level Comparisons
    report_lines.append("## Parcel-Level Comparisons")
    report_lines.append("")
    
    parcel_comparisons = results.get('parcel_comparisons')
    if parcel_comparisons is not None and len(parcel_comparisons) > 0:
        # Count significant parcels
        if 'pvalue_corrected' in parcel_comparisons.columns:
            fdr_sig = parcel_comparisons[parcel_comparisons.get('pvalue_corrected', pd.Series([False] * len(parcel_comparisons))) < 0.05]
            report_lines.append(f"**Total parcels tested:** {len(parcel_comparisons)}")
            report_lines.append(f"**Significant parcels (FDR corrected, p < 0.05):** {len(fdr_sig)}")
            report_lines.append("")
            
            if len(fdr_sig) > 0:
                report_lines.append("### Top 20 Most Significant Parcels (FDR corrected)")
                report_lines.append("")
                top_parcels = fdr_sig.nsmallest(20, 'pvalue_corrected')[['parcel_id', 'pvalue_corrected', 'statistic']]
                report_lines.append(_dataframe_to_markdown(top_parcels))
                report_lines.append("")
    
    # Spatial Analysis
    report_lines.append("## Spatial Analysis")
    report_lines.append("")
    
    spatial_analysis = results.get('spatial_analysis', {})
    if spatial_analysis:
        region_stats = spatial_analysis.get('region_stats')
        if region_stats is not None and len(region_stats) > 0:
            report_lines.append("### Regional Statistics")
            report_lines.append("")
            # Summary by region
            region_summary = region_stats.groupby('region_category').agg({
                'mean': 'mean',
                'std': 'mean',
                'n_parcels': 'first'
            }).reset_index()
            report_lines.append(_dataframe_to_markdown(region_summary))
            report_lines.append("")
        
        regional_tests = spatial_analysis.get('regional_tests')
        if regional_tests is not None and len(regional_tests) > 0:
            sig_regions = regional_tests[regional_tests['pvalue'] < 0.05]
            report_lines.append(f"**Significant regions (p < 0.05):** {len(sig_regions)}")
            if len(sig_regions) > 0:
                report_lines.append("")
                report_lines.append(_dataframe_to_markdown(sig_regions[['region_category', 'test', 'pvalue']]))
            report_lines.append("")
    
    # Figures
    report_lines.append("## Generated Figures")
    report_lines.append("")
    
    figures_dir = Path(results.get('output_dir', '')) / 'figures'
    
    figure_categories = {
        'Assumption Checks': 'assumption_checks',
        'Distributions': 'distributions',
        'Comparisons': 'comparisons',
        'Brain Projections': 'brain_projections'
    }
    
    for category, subdir in figure_categories.items():
        report_lines.append(f"### {category}")
        report_lines.append("")
        fig_dir = figures_dir / subdir
        if fig_dir.exists():
            fig_files = sorted(fig_dir.glob(f'sdi_{band}_*.png'))
            for fig_file in fig_files:
                rel_path = fig_file.relative_to(figures_dir.parent)
                report_lines.append(f"- [{fig_file.name}]({rel_path})")
        report_lines.append("")
    
    # Files
    report_lines.append("## Output Files")
    report_lines.append("")
    
    file_categories = {
        'Descriptive Statistics': 'descriptive_stats',
        'Statistical Tests': 'statistical_tests',
        'Reports': 'reports'
    }
    
    for category, subdir in file_categories.items():
        report_lines.append(f"### {category}")
        report_lines.append("")
        file_dir = Path(results.get('output_dir', '')) / subdir
        if file_dir.exists():
            csv_files = sorted(file_dir.glob(f'sdi_{band}_*.csv'))
            for csv_file in csv_files:
                rel_path = csv_file.relative_to(file_dir.parent)
                report_lines.append(f"- [{csv_file.name}]({rel_path})")
        report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Generated analysis report: {output_path}")

