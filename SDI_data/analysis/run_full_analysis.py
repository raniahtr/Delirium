"""
Main pipeline orchestrator for SDI analysis.

This module orchestrates all analysis modules and generates comprehensive reports.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Import analysis modules
# Handle both relative imports (when run as module) and absolute imports (when run directly)
try:
    from .load_sdi_data import load_all_sdi_data
    from .descriptive_stats import export_descriptive_stats
    from .assumption_checks import (
        check_all_assumptions, 
        export_assumption_checks,
        plot_qq_plots,
        plot_variance_comparison
    )
    from .group_comparisons import perform_group_comparisons, export_group_comparisons
    from .parcel_comparisons import (
        perform_parcel_level_comparisons,
        create_significance_map,
        create_continuous_significance_map,
        export_parcel_comparisons
    )
    from .spatial_analysis import (
        aggregate_by_region,
        aggregate_by_yeo7_network,
        export_spatial_analysis
    )
    from .visualization import (
        plot_distributions,
        plot_group_comparison,
        plot_brain_projections,
        plot_difference_maps,
        plot_significance_maps,
        plot_parcel_heatmap,
        plot_effect_sizes,
        plot_regional_comparison,
        plot_yeo7_network_comparison,
        plot_outlier_analysis
    )
except ImportError:
    # If relative imports fail, try absolute imports (when run directly)
    import sys
    from pathlib import Path
    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from analysis.load_sdi_data import load_all_sdi_data
    from analysis.descriptive_stats import export_descriptive_stats
    from analysis.assumption_checks import (
        check_all_assumptions, 
        export_assumption_checks,
        plot_qq_plots,
        plot_variance_comparison
    )
    from analysis.group_comparisons import perform_group_comparisons, export_group_comparisons
    from analysis.parcel_comparisons import (
        perform_parcel_level_comparisons,
        create_significance_map,
        create_continuous_significance_map,
        export_parcel_comparisons
    )
    from analysis.spatial_analysis import (
        aggregate_by_region,
        aggregate_by_yeo7_network,
        export_spatial_analysis
    )
    from analysis.visualization import (
        plot_distributions,
        plot_group_comparison,
        plot_brain_projections,
        plot_difference_maps,
        plot_significance_maps,
        plot_parcel_heatmap,
        plot_effect_sizes,
        plot_regional_comparison,
        plot_yeo7_network_comparison,
        plot_outlier_analysis
    )

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.load_config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sdi_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_full_analysis(
    config: Dict,
    band: str = 'bandpass',
    output_dir: Optional[str] = None,
    use_parametric: bool = False,
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh',
    generate_plots: bool = True,
    random_seed: int = 42
) -> Dict:
    """
    Run the complete SDI analysis pipeline.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    band : str
        Frequency band (default: 'bandpass')
    output_dir : str, optional
        Output directory. If None, uses results/analysis/
    use_parametric : bool
        Whether to use parametric tests (default: False)
    alpha : float
        Significance level (default: 0.05)
    correction_method : str
        Multiple comparisons correction method (default: 'fdr_bh')
    generate_plots : bool
        Whether to generate plots (default: True)
    random_seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    results : dict
        Dictionary with all analysis results
    """
    np.random.seed(random_seed)
    
    # Set output directory
    if output_dir is None:
        results_dir = config['dir']['results_dir']
        output_dir = os.path.join(results_dir, 'analysis')
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("SDI Analysis Pipeline")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frequency band: {band}")
    logger.info(f"Test type: {'Parametric' if use_parametric else 'Non-parametric'}")
    logger.info(f"Alpha level: {alpha}")
    logger.info(f"Correction method: {correction_method}")
    logger.info("=" * 80)
    
    results = {}
    
    # Step 1: Load data
    logger.info("\n[Step 1/8] Loading SDI data...")
    try:
        unified_df, group_arrays = load_all_sdi_data(config, band=band, use_individual_data=True)
        results['unified_df'] = unified_df
        results['group_arrays'] = group_arrays
        
        # Check if individual subject data was loaded
        has_individual_data = 'subject_id' in unified_df.columns
        if has_individual_data:
            n_subjects_total = unified_df['subject_id'].nunique()
            logger.info(f"✓ Loaded INDIVIDUAL SUBJECT data for {len(group_arrays)} groups")
            logger.info(f"  Total subjects: {n_subjects_total}, Total rows: {len(unified_df)}")
            for group in unified_df['group'].unique():
                n_subjects = unified_df[unified_df['group'] == group]['subject_id'].nunique()
                n_parcels = unified_df[unified_df['group'] == group]['parcel_id'].nunique()
                logger.info(f"  {group}: {n_subjects} subjects × {n_parcels} parcels")
            logger.info("  ✓ Statistical tests will be VALID (multiple subjects per group)")
        else:
            logger.info(f"✓ Loaded GROUP-AVERAGED data for {len(group_arrays)} groups, {len(unified_df)} total rows")
            logger.warning("  ⚠ Statistical tests will be INVALID (only one value per group)")
            logger.warning("  ⚠ Only descriptive analysis is meaningful")
        
        results['has_individual_data'] = has_individual_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # Step 2: Descriptive statistics
    logger.info("\n[Step 2/8] Computing descriptive statistics...")
    try:
        desc_stats_files = export_descriptive_stats(
            unified_df,
            output_dir=str(output_path / 'descriptive_stats'),
            prefix=f"sdi_{band}"
        )
        results['descriptive_stats'] = desc_stats_files
        logger.info("✓ Descriptive statistics computed and exported")
    except Exception as e:
        logger.error(f"Error computing descriptive statistics: {e}")
        results['descriptive_stats'] = None
    
    # Step 3: Assumption checks
    logger.info("\n[Step 3/8] Checking statistical assumptions...")
    try:
        assumption_results = check_all_assumptions(unified_df, alpha=alpha)
        assumption_files = export_assumption_checks(
            assumption_results,
            output_dir=str(output_path / 'statistical_tests'),
            prefix=f"sdi_{band}"
        )
        results['assumption_checks'] = assumption_results
        results['assumption_files'] = assumption_files
        
        # Generate assumption plots
        if generate_plots:
            qq_path = output_path / 'figures' / 'assumption_checks' / f'sdi_{band}_qq_plots.png'
            qq_path.parent.mkdir(parents=True, exist_ok=True)
            plot_qq_plots(unified_df, output_path=str(qq_path))
            
            var_path = output_path / 'figures' / 'assumption_checks' / f'sdi_{band}_variance_comparison.png'
            plot_variance_comparison(unified_df, output_path=str(var_path))
        
        # Update use_parametric based on assumptions
        if assumption_results['recommendations']['use_parametric']:
            logger.info("✓ Assumptions met - parametric tests recommended")
        else:
            logger.info("✓ Assumptions not met - non-parametric tests recommended")
            use_parametric = False
        
        logger.info("✓ Assumption checks completed")
    except Exception as e:
        logger.error(f"Error checking assumptions: {e}")
        results['assumption_checks'] = None
    
    # Step 4: Group-level comparisons
    logger.info("\n[Step 4/8] Performing group-level comparisons...")
    try:
        group_comparison_results = perform_group_comparisons(
            unified_df,
            use_parametric=use_parametric,
            correction_method=correction_method,
            alpha=alpha
        )
        group_files = export_group_comparisons(
            group_comparison_results,
            output_dir=str(output_path / 'statistical_tests'),
            prefix=f"sdi_{band}"
        )
        results['group_comparisons'] = group_comparison_results
        results['group_files'] = group_files
        logger.info("✓ Group-level comparisons completed")
    except Exception as e:
        logger.error(f"Error performing group comparisons: {e}")
        results['group_comparisons'] = None
    
    # Step 5: Parcel-level comparisons
    logger.info("\n[Step 5/8] Performing parcel-level comparisons...")
    try:
        parcel_results = perform_parcel_level_comparisons(
            unified_df,
            test_type='anova' if use_parametric else 'kruskal_wallis',
            posthoc_test='ttest' if use_parametric else 'mannwhitneyu',
            correction_methods=['fdr_bh', 'bonferroni'],
            alpha=alpha
        )
        parcel_files = export_parcel_comparisons(
            parcel_results,
            output_dir=str(output_path / 'statistical_tests'),
            prefix=f"sdi_{band}"
        )
        results['parcel_comparisons'] = parcel_results
        results['parcel_files'] = parcel_files
        
        # Create significance maps
        sig_map_fdr = create_significance_map(parcel_results, 'fdr_bh', alpha)
        sig_map_cont = create_continuous_significance_map(parcel_results, 'fdr_bh')
        results['significance_map_fdr'] = sig_map_fdr
        results['significance_map_continuous'] = sig_map_cont
        
        logger.info(f"✓ Parcel-level comparisons completed: {sig_map_fdr.sum()} significant parcels (FDR)")
    except Exception as e:
        logger.error(f"Error performing parcel comparisons: {e}")
        results['parcel_comparisons'] = None
    
    # Step 6: Spatial analysis
    logger.info("\n[Step 6/8] Performing spatial analysis...")
    try:
        region_stats = aggregate_by_region(unified_df, config=config)
        network_stats = aggregate_by_yeo7_network(unified_df, config=config)
        spatial_files = export_spatial_analysis(
            region_stats,
            network_stats,
            output_dir=str(output_path / 'statistical_tests'),
            prefix=f"sdi_{band}"
        )
        results['spatial_analysis'] = {
            'regional': region_stats,
            'yeo7_networks': network_stats
        }
        results['spatial_files'] = spatial_files
        logger.info("✓ Spatial analysis completed")
    except Exception as e:
        logger.error(f"Error performing spatial analysis: {e}")
        results['spatial_analysis'] = None
    
    # Step 7: Visualizations
    if generate_plots:
        logger.info("\n[Step 7/8] Generating visualizations...")
        try:
            # Distribution plots (combined violin+boxplot)
            dist_path = output_path / 'figures' / 'distributions' / f'sdi_{band}_distributions.png'
            dist_path.parent.mkdir(parents=True, exist_ok=True)
            plot_distributions(
                unified_df, 
                output_path=str(dist_path),
                combine_plots=True,
                use_log_scale=True,
                percentile_clip=(1, 99)
            )
            
            # Group comparison plot
            posthoc_df = results.get('group_comparisons', {}).get('posthoc_tests', pd.DataFrame())
            comp_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_group_comparison.png'
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            plot_group_comparison(
                unified_df, 
                posthoc_df, 
                output_path=str(comp_path),
                use_log_scale=True,
                percentile_clip=(1, 99)
            )
            
            # Brain projections (with consistent scaling)
            brain_path = output_path / 'figures' / 'brain_projections'
            brain_path.mkdir(parents=True, exist_ok=True)
            brain_files = plot_brain_projections(
                group_arrays,
                config,
                str(brain_path),
                prefix=f"sdi_{band}",
                use_consistent_scale=True,
                use_log2=True
            )
            results['brain_projection_files'] = brain_files
            
            # Get significance maps for difference maps
            significance_maps = {}
            if 'parcel_comparisons' in results and results['parcel_comparisons'] is not None:
                from .parcel_comparisons import create_significance_map
                parcel_results = pd.read_csv(results['parcel_comparisons']['parcel_results'])
                sig_map_fdr = create_significance_map(parcel_results, correction_method='fdr_bh')
                # Create significance maps for each comparison
                groups = sorted(group_arrays.keys())
                for i, g1 in enumerate(groups):
                    for g2 in groups[i+1:]:
                        comp_name = f"{g1.replace(' ', '_')}_vs_{g2.replace(' ', '_')}"
                        significance_maps[comp_name] = sig_map_fdr
            
            # Difference maps (with significance overlay and log transform)
            diff_files = plot_difference_maps(
                group_arrays,
                config,
                str(brain_path),
                prefix=f"sdi_{band}",
                significance_maps=significance_maps if len(significance_maps) > 0 else None,
                use_log_transform=True,
                use_percent_change=False
            )
            results['difference_map_files'] = diff_files
            
            # Significance maps
            if 'significance_map_fdr' in results:
                sig_path = output_path / 'figures' / 'significance_maps' / f'sdi_{band}_significance_fdr.png'
                sig_path.parent.mkdir(parents=True, exist_ok=True)
                plot_significance_maps(
                    results['significance_map_fdr'],
                    config,
                    str(sig_path),
                    title="Significant Parcels (FDR-corrected)"
                )
            
            # Parcel heatmap (with log transform, spatial ordering, significance overlay)
            parcel_stats = pd.read_csv(results['descriptive_stats']['parcel_level'])
            heatmap_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_parcel_heatmap.png'
            sig_map_for_heatmap = None
            if 'parcel_comparisons' in results and results['parcel_comparisons'] is not None:
                from .parcel_comparisons import create_significance_map
                parcel_results = pd.read_csv(results['parcel_comparisons']['parcel_results'])
                sig_map_for_heatmap = create_significance_map(parcel_results, correction_method='fdr_bh')
            plot_parcel_heatmap(
                parcel_stats, 
                output_path=str(heatmap_path),
                config=config,
                significance_map=sig_map_for_heatmap,
                use_log_scale=True,
                use_median=False  # Use mean for now (will use median when individual data available)
            )
            
            # Effect sizes (always generate with interpretation guide)
            effect_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_effect_sizes.png'
            plot_effect_sizes(
                posthoc_df, 
                output_path=str(effect_path),
                always_generate=True
            )
            
            # Regional comparison (with median, log scale, error bars)
            if 'spatial_analysis' in results and results['spatial_analysis'] is not None:
                reg_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_regional_comparison.png'
                plot_regional_comparison(
                    region_stats, 
                    output_path=str(reg_path),
                    use_median=True,
                    use_log_scale=True,
                    show_error_bars=True
                )
            
            # Yeo7 network comparison
            if 'spatial_analysis' in results and results['spatial_analysis'] is not None:
                network_stats_path = results['spatial_analysis'].get('network_stats')
                if network_stats_path and Path(network_stats_path).exists():
                    network_stats = pd.read_csv(network_stats_path)
                    network_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_yeo7_network_comparison.png'
                    plot_yeo7_network_comparison(
                        network_stats,
                        output_path=str(network_path),
                        use_median=True,
                        use_log_scale=True,
                        show_error_bars=True
                    )
            
            # Outlier analysis
            outlier_path = output_path / 'figures' / 'distributions' / f'sdi_{band}_outlier_analysis.png'
            plot_outlier_analysis(
                unified_df,
                output_path=str(outlier_path)
            )
            
            logger.info("✓ Visualizations generated")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    # Step 8: Generate report
    logger.info("\n[Step 8/8] Generating analysis report...")
    try:
        report_path = generate_analysis_report(
            results,
            output_dir=str(output_path / 'reports'),
            prefix=f"sdi_{band}"
        )
        results['report_path'] = report_path
        logger.info(f"✓ Analysis report generated: {report_path}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        results['report_path'] = None
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis pipeline completed successfully!")
    logger.info("=" * 80)
    
    return results


def generate_analysis_report(
    results: Dict,
    output_dir: str,
    prefix: str = "sdi"
) -> str:
    """
    Generate a comprehensive markdown report of the analysis.
    
    Parameters:
    -----------
    results : dict
        Dictionary with all analysis results
    output_dir : str
        Output directory
    prefix : str
        File prefix
    
    Returns:
    --------
    report_path : str
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / f"{prefix}_analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# SDI Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        if 'group_comparisons' in results and results['group_comparisons']:
            primary_test = results['group_comparisons']['primary_test']
            f.write(f"- Primary test: {primary_test['test']}\n")
            f.write(f"- Test statistic: {primary_test['statistic']:.4f}\n")
            f.write(f"- P-value: {primary_test['pvalue']:.6f}\n")
            f.write(f"- Significant: {'Yes' if primary_test['significant'] else 'No'}\n\n")
        
        if 'parcel_comparisons' in results and results['parcel_comparisons'] is not None:
            parcel_df = results['parcel_comparisons']
            n_sig_fdr = parcel_df['significant_fdr_bh'].sum() if 'significant_fdr_bh' in parcel_df.columns else 0
            n_sig_bonf = parcel_df['significant_bonferroni'].sum() if 'significant_bonferroni' in parcel_df.columns else 0
            f.write(f"- Significant parcels (FDR): {n_sig_fdr}\n")
            f.write(f"- Significant parcels (Bonferroni): {n_sig_bonf}\n\n")
        
        # File locations
        f.write("## Output Files\n\n")
        f.write("### Descriptive Statistics\n")
        if 'descriptive_stats' in results and results['descriptive_stats']:
            for key, path in results['descriptive_stats'].items():
                if path:
                    f.write(f"- {key}: `{path}`\n")
        f.write("\n")
        
        f.write("### Statistical Tests\n")
        if 'group_files' in results and results['group_files']:
            for key, path in results['group_files'].items():
                if path:
                    f.write(f"- {key}: `{path}`\n")
        if 'parcel_files' in results and results['parcel_files']:
            for key, path in results['parcel_files'].items():
                if path:
                    f.write(f"- {key}: `{path}`\n")
        f.write("\n")
        
        f.write("### Figures\n")
        f.write("See `figures/` directory for all generated plots.\n\n")
        
        f.write("---\n\n")
        f.write("## Detailed Results\n\n")
        f.write("See individual CSV files for detailed statistical results.\n")
    
    return str(report_path)


def main():
    """Command-line interface for running the analysis pipeline."""
    parser = argparse.ArgumentParser(description='Run SDI analysis pipeline')
    parser.add_argument('--band', type=str, default='bandpass', help='Frequency band')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--parametric', action='store_true', help='Use parametric tests')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--correction', type=str, default='fdr_bh', help='Correction method')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Run analysis
    results = run_full_analysis(
        config=config,
        band=args.band,
        output_dir=args.output_dir,
        use_parametric=args.parametric,
        alpha=args.alpha,
        correction_method=args.correction,
        generate_plots=not args.no_plots,
        random_seed=args.seed
    )
    
    print(f"\nAnalysis complete! Results saved to: {results.get('report_path', 'N/A')}")


if __name__ == "__main__":
    main()

