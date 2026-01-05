"""
Example script demonstrating how to run the SDI analysis pipeline.

This script shows different ways to use the analysis modules:
1. Running the full pipeline
2. Running individual modules
3. Customizing analysis parameters
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.load_config import load_config
from analysis.run_full_analysis import run_full_analysis
from analysis.load_sdi_data import load_all_sdi_data
from analysis.descriptive_stats import compute_group_level_stats, export_descriptive_stats
from analysis.assumption_checks import check_all_assumptions
from analysis.group_comparisons import perform_group_comparisons
from analysis.parcel_comparisons import perform_parcel_level_comparisons
from analysis.visualization import plot_distributions, plot_group_comparison


def example_full_pipeline():
    """Example 1: Run the complete analysis pipeline."""
    print("=" * 80)
    print("Example 1: Running Full Analysis Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Run full pipeline with default settings
    results = run_full_analysis(
        config=config,
        band='allFB',
        use_parametric=False,  # Use non-parametric tests (recommended)
        alpha=0.05,
        correction_method='fdr_bh',
        generate_plots=True,
        random_seed=42
    )
    
    print(f"\n✓ Full pipeline completed!")
    print(f"  Report saved to: {results.get('report_path', 'N/A')}")
    print(f"  Significant parcels (FDR): {results.get('significance_map_fdr', []).sum() if 'significance_map_fdr' in results else 'N/A'}")


def example_individual_modules():
    """Example 2: Run individual analysis modules."""
    print("\n" + "=" * 80)
    print("Example 2: Running Individual Modules")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Step 1: Load data
    print("\n1. Loading SDI data...")
    unified_df, group_arrays = load_all_sdi_data(config, band='allFB')
    print(f"   ✓ Loaded {len(group_arrays)} groups, {len(unified_df)} total rows")
    
    # Step 2: Descriptive statistics
    print("\n2. Computing descriptive statistics...")
    group_stats = compute_group_level_stats(unified_df)
    print("\n   Group-level statistics:")
    print(group_stats[['group', 'mean', 'median', 'std']].to_string(index=False))
    
    # Step 3: Assumption checks
    print("\n3. Checking assumptions...")
    assumption_results = check_all_assumptions(unified_df)
    recommendations = assumption_results['recommendations']
    print(f"   Use parametric tests: {recommendations['use_parametric']}")
    print(f"   Recommended primary test: {recommendations['primary_test']}")
    
    # Step 4: Group comparisons
    print("\n4. Performing group comparisons...")
    use_parametric = recommendations['use_parametric']
    group_results = perform_group_comparisons(
        unified_df,
        use_parametric=use_parametric,
        correction_method='fdr_bh',
        alpha=0.05
    )
    
    primary_test = group_results['primary_test']
    print(f"   Primary test: {primary_test['test']}")
    print(f"   P-value: {primary_test['pvalue']:.6f}")
    print(f"   Significant: {primary_test['significant']}")
    
    if len(group_results['posthoc_tests']) > 0:
        print("\n   Post-hoc test results:")
        print(group_results['posthoc_tests'][['group1', 'group2', 'pvalue_corrected', 'effect_size']].to_string(index=False))
    
    # Step 5: Parcel-level comparisons (sample - first 10 parcels for speed)
    print("\n5. Performing parcel-level comparisons (first 10 parcels as example)...")
    sample_df = unified_df[unified_df['parcel_id'] < 10]  # Just first 10 parcels
    parcel_results = perform_parcel_level_comparisons(
        sample_df,
        test_type='kruskal_wallis',
        posthoc_test='mannwhitneyu',
        correction_methods=['fdr_bh'],
        alpha=0.05
    )
    print(f"   ✓ Analyzed {len(parcel_results)} parcels")
    if 'significant_fdr_bh' in parcel_results.columns:
        n_sig = parcel_results['significant_fdr_bh'].sum()
        print(f"   Significant parcels: {n_sig}")


def example_custom_analysis():
    """Example 3: Custom analysis with specific parameters."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Analysis")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Load data
    unified_df, group_arrays = load_all_sdi_data(config, band='allFB')
    
    # Custom: Compare only two specific groups
    print("\nComparing ICU vs Healthy Controls only...")
    groups_to_compare = ['ICU', 'Healthy Controls']
    filtered_df = unified_df[unified_df['group'].isin(groups_to_compare)]
    
    # Perform comparison
    group_results = perform_group_comparisons(
        filtered_df,
        use_parametric=False,
        correction_method='fdr_bh',
        alpha=0.05
    )
    
    print(f"Primary test: {group_results['primary_test']['test']}")
    print(f"P-value: {group_results['primary_test']['pvalue']:.6f}")
    
    if len(group_results['posthoc_tests']) > 0:
        posthoc = group_results['posthoc_tests'].iloc[0]
        print(f"\nPost-hoc comparison:")
        print(f"  {posthoc['group1']} vs {posthoc['group2']}")
        print(f"  Effect size ({posthoc['effect_size_name']}): {posthoc['effect_size']:.4f}")
        print(f"  P-value (FDR-corrected): {posthoc['pvalue_corrected']:.6f}")


def example_visualization_only():
    """Example 4: Generate visualizations only."""
    print("\n" + "=" * 80)
    print("Example 4: Visualization Only")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Load data
    unified_df, group_arrays = load_all_sdi_data(config, band='allFB')
    
    # Generate distribution plots
    print("\nGenerating distribution plots...")
    plot_distributions(
        unified_df,
        output_path='example_distributions.png'
    )
    print("   ✓ Saved to: example_distributions.png")
    
    # Generate group comparison plot
    print("\nGenerating group comparison plot...")
    group_results = perform_group_comparisons(unified_df, use_parametric=False)
    posthoc_df = group_results.get('posthoc_tests', pd.DataFrame()) if group_results else pd.DataFrame()
    
    plot_group_comparison(
        unified_df,
        posthoc_results=posthoc_df,
        output_path='example_group_comparison.png'
    )
    print("   ✓ Saved to: example_group_comparison.png")


if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("SDI Analysis Pipeline - Example Scripts")
    print("=" * 80)
    print("\nThis script demonstrates different ways to use the analysis pipeline.")
    print("Uncomment the example you want to run.\n")
    
    # Uncomment the example you want to run:
    
    # Example 1: Full pipeline (comprehensive, takes longer)
    # example_full_pipeline()
    
    # Example 2: Individual modules (step-by-step)
    example_individual_modules()
    
    # Example 3: Custom analysis
    # example_custom_analysis()
    
    # Example 4: Visualization only
    # example_visualization_only()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)

