"""
Main analysis pipeline orchestrator for SDI data.

This module runs the complete analysis pipeline in sequence.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Import analysis modules
from analysis.load_sdi_data import load_sdi_hc_icu_data, load_atlas_to_yeo7_mapping
from analysis.descriptive_stats import export_descriptive_stats
from analysis.assumption_checks import check_all_assumptions, export_assumption_checks, plot_qq_plots, plot_variance_comparison
from analysis.group_comparisons import perform_group_comparisons, export_group_comparisons
from analysis.parcel_comparisons import perform_parcel_level_comparisons, export_parcel_comparisons
from analysis.spatial_analysis import aggregate_by_region, perform_regional_statistical_tests, export_regional_stats
from analysis.visualization import (
    plot_distributions, plot_group_comparison, plot_effect_sizes,
    plot_regional_comparison, plot_parcel_heatmap, plot_brain_projections, plot_difference_maps
)
from analysis.report_generator import generate_analysis_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis(
    config: Dict,
    band: str = 'bandpass',
    output_dir: Optional[str] = None,
    use_parametric: bool = False,
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh',
    generate_plots: bool = True,
    pipeline_type: str = 'HC_ICU'
) -> Dict:
    """
    Run complete SDI analysis pipeline.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    band : str
        Frequency band (default: 'bandpass')
    output_dir : str, optional
        Output directory (default: results/analysis)
    use_parametric : bool
        Whether to use parametric tests (default: False)
    alpha : float
        Significance level (default: 0.05)
    correction_method : str
        Multiple comparisons correction method (default: 'fdr_bh')
    generate_plots : bool
        Generate visualization plots (default: True)
    pipeline_type : str
        Pipeline type (default: 'HC_ICU')
    
    Returns:
    --------
    results : dict
        Dictionary with all analysis results
    """
    # Setup output directory
    if output_dir is None:
        results_dir = Path(config.get('results_dir', 'results'))
        output_path = results_dir / 'analysis'
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("SDI Analysis Pipeline")
    logger.info("=" * 80)
    logger.info(f"Band: {band}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Use parametric: {use_parametric}")
    logger.info(f"Alpha: {alpha}")
    logger.info(f"Correction method: {correction_method}")
    
    results = {
        'band': band,
        'output_dir': str(output_path),
        'config': config
    }
    
    try:
        # Step 1: Load data
        logger.info("\n[Step 1/8] Loading SDI data...")
        unified_df, group_arrays = load_sdi_hc_icu_data(
            config,
            band=band,
            pipeline_type=pipeline_type
        )
        
        if len(unified_df) == 0:
            raise ValueError(
                "No data loaded! This usually means group assignments could not be matched.\n"
                "Please check:\n"
                "1. Config has 'group' key pointing to Excel file\n"
                "2. Excel file has 'sub_ID' (or similar) and 'Group' columns\n"
                "3. Subject IDs in SDI file match those in Excel file (format: sub-01, sub-02, etc.)"
            )
        
        results['unified_df'] = unified_df
        results['group_arrays'] = group_arrays
        logger.info(f"✓ Loaded data: {len(unified_df)} rows, {len(group_arrays)} groups")
        
        # Load atlas mapping
        atlas_to_yeo7_df = load_atlas_to_yeo7_mapping()
        results['atlas_to_yeo7_df'] = atlas_to_yeo7_df
        
        # Step 2: Descriptive statistics
        logger.info("\n[Step 2/8] Computing descriptive statistics...")
        desc_stats_files = export_descriptive_stats(
            unified_df,
            output_dir=str(output_path / 'descriptive_stats'),
            prefix=f"sdi_{band}"
        )
        results['descriptive_stats'] = desc_stats_files
        logger.info("✓ Descriptive statistics computed and exported")
        
        # Step 3: Assumption checks
        logger.info("\n[Step 3/8] Checking statistical assumptions...")
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
        
        # Step 4: Group-level comparisons
        logger.info("\n[Step 4/8] Performing group-level comparisons...")
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
        
        # Step 5: Parcel-level comparisons
        logger.info("\n[Step 5/8] Performing parcel-level comparisons...")
        parcel_results_df = perform_parcel_level_comparisons(
            unified_df,
            use_parametric=use_parametric,
            alpha=alpha
        )
        parcel_files = export_parcel_comparisons(
            parcel_results_df,
            output_dir=str(output_path / 'statistical_tests'),
            prefix=f"sdi_{band}",
            apply_corrections=True,
            alpha=alpha
        )
        results['parcel_comparisons'] = parcel_results_df
        results['parcel_files'] = parcel_files
        logger.info("✓ Parcel-level comparisons completed")
        
        # Step 6: Spatial analysis
        logger.info("\n[Step 6/8] Performing spatial analysis...")
        region_stats_df = aggregate_by_region(unified_df, atlas_to_yeo7_df)
        regional_tests_df = perform_regional_statistical_tests(
            unified_df,
            atlas_to_yeo7_df,
            use_parametric=use_parametric
        )
        regional_files = export_regional_stats(
            region_stats_df,
            regional_tests_df,
            output_dir=str(output_path / 'statistical_tests'),
            prefix=f"sdi_{band}"
        )
        results['spatial_analysis'] = {
            'region_stats': region_stats_df,
            'regional_tests': regional_tests_df
        }
        results['regional_files'] = regional_files
        logger.info("✓ Spatial analysis completed")
        
        # Step 7: Generate visualizations
        if generate_plots:
            logger.info("\n[Step 7/8] Generating visualizations...")
            
            # Distribution plots
            dist_path = output_path / 'figures' / 'distributions' / f'sdi_{band}_distributions.png'
            dist_path.parent.mkdir(parents=True, exist_ok=True)
            plot_distributions(unified_df, output_path=str(dist_path))
            
            # Group comparison plot
            posthoc_df = group_comparison_results.get('posthoc_tests', pd.DataFrame())
            comp_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_group_comparison.png'
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            plot_group_comparison(unified_df, posthoc_df, output_path=str(comp_path))
            
            # Effect sizes
            if len(posthoc_df) > 0:
                effect_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_effect_sizes.png'
                plot_effect_sizes(posthoc_df, output_path=str(effect_path))
            
            # Regional comparison
            if len(region_stats_df) > 0:
                reg_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_regional_comparison.png'
                plot_regional_comparison(region_stats_df, output_path=str(reg_path))
            
            # Parcel heatmap
            parcel_heat_path = output_path / 'figures' / 'comparisons' / f'sdi_{band}_parcel_heatmap.png'
            plot_parcel_heatmap(parcel_results_df, output_path=str(parcel_heat_path))
            
            # Brain projections
            brain_path = output_path / 'figures' / 'brain_projections'
            brain_path.mkdir(parents=True, exist_ok=True)
            brain_files = plot_brain_projections(
                group_arrays,
                config,
                str(brain_path),
                prefix=f"sdi_{band}",
                use_log2=True
            )
            results['brain_projection_files'] = brain_files
            
            # Difference maps
            diff_files = plot_difference_maps(
                group_arrays,
                config,
                str(brain_path),
                prefix=f"sdi_{band}",
                use_log2=True
            )
            results['difference_map_files'] = diff_files
            
            logger.info("✓ Visualizations generated")
        
        # Step 8: Generate report
        logger.info("\n[Step 8/8] Generating analysis report...")
        report_path = output_path / 'reports' / f'sdi_{band}_analysis_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_analysis_report(
            results,
            output_path=str(report_path),
            band=band
        )
        results['report_path'] = str(report_path)
        logger.info(f"✓ Analysis report saved to {report_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis pipeline completed successfully!")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    from utils.load_config import load_config
    
    parser = argparse.ArgumentParser(
        description='Run SDI analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m analysis.run_analysis --band bandpass
  python -m analysis.run_analysis --band allFB --use-parametric
  python -m analysis.run_analysis --band bandpass --alpha 0.01 --correction bonferroni
        """
    )
    
    parser.add_argument(
        '--band',
        type=str,
        default='bandpass',
        help='Frequency band to analyze (default: bandpass)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: results/analysis)'
    )
    
    parser.add_argument(
        '--use-parametric',
        action='store_true',
        help='Use parametric tests (ANOVA, t-test) instead of non-parametric (default: False)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    parser.add_argument(
        '--correction',
        type=str,
        choices=['fdr_bh', 'bonferroni'],
        default='fdr_bh',
        help='Multiple comparisons correction method (default: fdr_bh)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (default: False, plots are generated)'
    )
    
    parser.add_argument(
        '--pipeline-type',
        type=str,
        default='HC_ICU',
        choices=['HC_ICU'],
        help='Pipeline type (default: HC_ICU)'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        print(f"Error: Could not load configuration file. Make sure config.json exists.")
        exit(1)
    
    # Run analysis
    try:
        results = run_analysis(
            config=config,
            band=args.band,
            output_dir=args.output_dir,
            use_parametric=args.use_parametric,
            alpha=args.alpha,
            correction_method=args.correction,
            generate_plots=not args.no_plots,
            pipeline_type=args.pipeline_type
        )
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        print(f"Results saved to: {results['output_dir']}")
        print(f"\nKey outputs:")
        print(f"  - Descriptive stats: {results['output_dir']}/descriptive_stats/")
        print(f"  - Statistical tests: {results['output_dir']}/statistical_tests/")
        print(f"  - Figures: {results['output_dir']}/figures/")
        print(f"  - Report: {results['output_dir']}/reports/sdi_{args.band}_analysis_report.md")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nError: Analysis failed. Check logs for details.")
        exit(1)

