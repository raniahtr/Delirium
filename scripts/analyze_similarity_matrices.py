#!/usr/bin/env python3
"""
Analyze subject similarity matrices from sc_similarity_matrices.h5

This script extracts numerical similarity statistics from the HDF5 file
and provides quantitative, data-driven interpretation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

ROOT_DIR = Path("/media/RCPNAS/Data/Delirium/Delirium_Rania")
PREPROC_DIR = ROOT_DIR / "Preproc_current"
SIMILARITY_FILE = PREPROC_DIR / "results" / "sc_similarity_matrices.h5"

def analyze_similarity_matrices(similarity_file: Path):
    """Extract and analyze similarity statistics from HDF5 file."""
    
    print("=" * 80)
    print("SUBJECT SIMILARITY ANALYSIS")
    print("=" * 80)
    print(f"\nLoading similarity matrices from: {similarity_file}")
    
    if not similarity_file.exists():
        print(f"ERROR: File not found: {similarity_file}")
        return
    
    # Load DataFrame
    df = pd.read_hdf(similarity_file, key='similarity_matrices')
    
    print(f"✓ Loaded {len(df)} similarity pairs")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Groups: {df['Group'].unique()}")
    print(f"  Region Categories: {df['Region Category'].unique()}")
    
    # Filter for SC_sift2_sizecorr if connectome_type column exists
    if 'Connectome Type' in df.columns:
        df = df[df['Connectome Type'] == 'SC_sift2_sizecorr']
        print(f"  Filtered to SC_sift2_sizecorr: {len(df)} pairs")
    
    # Group statistics
    print("\n" + "=" * 80)
    print("SIMILARITY STATISTICS BY GROUP AND REGION")
    print("=" * 80)
    
    groups = ['ICU', 'ICU Delirium', 'Healthy Controls']
    region_categories = sorted(df['Region Category'].unique())
    
    results_summary = []
    
    for group in groups:
        group_df = df[df['Group'] == group]
        if len(group_df) == 0:
            continue
        
        print(f"\n{'=' * 80}")
        print(f"GROUP: {group}")
        print(f"{'=' * 80}")
        print(f"Total pairs: {len(group_df)}")
        
        # Overall statistics
        overall_mean = group_df['Similarity'].mean()
        overall_std = group_df['Similarity'].std()
        overall_median = group_df['Similarity'].median()
        overall_min = group_df['Similarity'].min()
        overall_max = group_df['Similarity'].max()
        
        print(f"\nOverall Similarity (all regions):")
        print(f"  Mean:   {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"  Median: {overall_median:.4f}")
        print(f"  Range:  [{overall_min:.4f}, {overall_max:.4f}]")
        
        results_summary.append({
            'Group': group,
            'Region': 'Overall',
            'Mean': overall_mean,
            'Std': overall_std,
            'Median': overall_median,
            'Min': overall_min,
            'Max': overall_max,
            'N_pairs': len(group_df)
        })
        
        # By region category
        print(f"\nBy Region Category:")
        for region in region_categories:
            region_df = group_df[group_df['Region Category'] == region]
            if len(region_df) == 0:
                continue
            
            region_mean = region_df['Similarity'].mean()
            region_std = region_df['Similarity'].std()
            region_median = region_df['Similarity'].median()
            region_min = region_df['Similarity'].min()
            region_max = region_df['Similarity'].max()
            
            print(f"  {region:25s}: Mean={region_mean:7.4f} ± {region_std:6.4f}, "
                  f"Median={region_median:7.4f}, Range=[{region_min:6.4f}, {region_max:6.4f}], "
                  f"N={len(region_df)}")
            
            results_summary.append({
                'Group': group,
                'Region': region,
                'Mean': region_mean,
                'Std': region_std,
                'Median': region_median,
                'Min': region_min,
                'Max': region_max,
                'N_pairs': len(region_df)
            })
    
    # Between-group comparisons
    print("\n" + "=" * 80)
    print("BETWEEN-GROUP COMPARISONS")
    print("=" * 80)
    
    from scipy import stats
    
    comparison_results = []
    
    for region in ['Overall'] + list(region_categories):
        if region == 'Overall':
            region_df = df
        else:
            region_df = df[df['Region Category'] == region]
        
        if len(region_df) == 0:
            continue
        
        print(f"\n{region}:")
        
        # Get similarity values for each group
        group_values = {}
        for group in groups:
            group_vals = region_df[region_df['Group'] == group]['Similarity'].values
            if len(group_vals) > 0:
                group_values[group] = group_vals
        
        # Pairwise comparisons
        for i, group1 in enumerate(groups):
            if group1 not in group_values:
                continue
            for group2 in groups[i+1:]:
                if group2 not in group_values:
                    continue
                
                vals1 = group_values[group1]
                vals2 = group_values[group2]
                
                # Statistical test
                # Check normality
                _, p_norm1 = stats.shapiro(vals1) if len(vals1) <= 5000 else (None, 0.05)
                _, p_norm2 = stats.shapiro(vals2) if len(vals2) <= 5000 else (None, 0.05)
                
                if p_norm1 > 0.05 and p_norm2 > 0.05 and len(vals1) > 2 and len(vals2) > 2:
                    # Use t-test
                    stat, p_val = stats.ttest_ind(vals1, vals2)
                    test_name = "t-test"
                else:
                    # Use Mann-Whitney U
                    stat, p_val = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    test_name = "Mann-Whitney U"
                
                mean1 = np.mean(vals1)
                mean2 = np.mean(vals2)
                median1 = np.median(vals1)
                median2 = np.median(vals2)
                
                print(f"  {group1:20s} vs {group2:20s}:")
                print(f"    Mean:   {mean1:.4f} vs {mean2:.4f} (diff: {mean1-mean2:+.4f})")
                print(f"    Median: {median1:.4f} vs {median2:.4f} (diff: {median1-median2:+.4f})")
                print(f"    {test_name}: stat={stat:.4f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
                
                comparison_results.append({
                    'Region': region,
                    'Group1': group1,
                    'Group2': group2,
                    'Mean1': mean1,
                    'Mean2': mean2,
                    'Mean_diff': mean1 - mean2,
                    'Median1': median1,
                    'Median2': median2,
                    'Median_diff': median1 - median2,
                    'Test': test_name,
                    'Statistic': stat,
                    'P_value': p_val,
                    'N1': len(vals1),
                    'N2': len(vals2)
                })
    
    # Save summary tables
    summary_df = pd.DataFrame(results_summary)
    comparison_df = pd.DataFrame(comparison_results)
    
    output_dir = PREPROC_DIR / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    summary_file = output_dir / "similarity_statistics_summary.csv"
    comparison_file = output_dir / "similarity_comparisons.csv"
    
    summary_df.to_csv(summary_file, index=False)
    comparison_df.to_csv(comparison_file, index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Saved summary statistics to: {summary_file}")
    print(f"✓ Saved between-group comparisons to: {comparison_file}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Overall similarity by group
    overall_stats = summary_df[summary_df['Region'] == 'Overall']
    if len(overall_stats) > 0:
        print("\nOverall Similarity (all regions combined):")
        for _, row in overall_stats.iterrows():
            print(f"  {row['Group']:20s}: Mean = {row['Mean']:.4f} ± {row['Std']:.4f}, "
                  f"Median = {row['Median']:.4f}, N = {row['N_pairs']} pairs")
    
    # Significant comparisons
    sig_comparisons = comparison_df[comparison_df['P_value'] < 0.05]
    if len(sig_comparisons) > 0:
        print("\nSignificant Between-Group Differences (p < 0.05):")
        for _, row in sig_comparisons.iterrows():
            print(f"  {row['Region']:25s}: {row['Group1']:20s} vs {row['Group2']:20s}")
            print(f"    Mean difference: {row['Mean_diff']:+.4f}, p = {row['P_value']:.6f}")
    else:
        print("\nNo significant between-group differences found (p < 0.05)")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if len(overall_stats) >= 3:
        hc_mean = overall_stats[overall_stats['Group'] == 'Healthy Controls']['Mean'].values[0]
        icu_mean = overall_stats[overall_stats['Group'] == 'ICU']['Mean'].values[0] if len(overall_stats[overall_stats['Group'] == 'ICU']) > 0 else None
        delirium_mean = overall_stats[overall_stats['Group'] == 'ICU Delirium']['Mean'].values[0]
        
        print(f"\n1. Overall Similarity:")
        print(f"   - Healthy Controls: {hc_mean:.4f}")
        if icu_mean is not None:
            print(f"   - ICU: {icu_mean:.4f} (diff from HC: {icu_mean-hc_mean:+.4f})")
        print(f"   - ICU Delirium: {delirium_mean:.4f} (diff from HC: {delirium_mean-hc_mean:+.4f})")
        
        if icu_mean is not None:
            print(f"\n2. ICU vs ICU Delirium:")
            print(f"   - Difference: {delirium_mean-icu_mean:+.4f}")
            icu_del_comparison = comparison_df[
                (comparison_df['Group1'] == 'ICU') & 
                (comparison_df['Group2'] == 'ICU Delirium') &
                (comparison_df['Region'] == 'Overall')
            ]
            if len(icu_del_comparison) > 0:
                p_val = icu_del_comparison['P_value'].values[0]
                print(f"   - Statistical significance: p = {p_val:.6f} {'(significant)' if p_val < 0.05 else '(not significant)'}")
        
        print(f"\n3. Interpretation:")
        print(f"   - Higher similarity = more similar connectome patterns within group")
        print(f"   - Lower similarity = more variable connectome patterns within group")
        if hc_mean > delirium_mean:
            print(f"   - Healthy Controls show {'higher' if hc_mean > delirium_mean else 'lower'} within-group similarity")
            print(f"     than ICU Delirium, suggesting {'more consistent' if hc_mean > delirium_mean else 'more variable'} patterns")
    
    return summary_df, comparison_df

if __name__ == "__main__":
    summary_df, comparison_df = analyze_similarity_matrices(SIMILARITY_FILE)
    print("\n✓ Analysis complete!")

