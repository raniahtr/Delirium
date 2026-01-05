"""
Parcel-level statistical comparisons for SDI data.

This module performs statistical tests for each of the 430 parcels,
with multiple comparisons correction (FDR, Bonferroni).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def compare_parcel_across_groups(
    parcel_data: Dict[str, np.ndarray],
    test_type: str = 'kruskal_wallis',
    posthoc_test: str = 'mannwhitneyu'
) -> Dict:
    """
    Compare SDI values for a single parcel across groups.
    
    Parameters:
    -----------
    parcel_data : dict
        Dictionary mapping group names to SDI values for this parcel
    test_type : str
        'kruskal_wallis' or 'anova'
    posthoc_test : str
        'mannwhitneyu' or 'ttest'
    
    Returns:
    --------
    results : dict
        Test results for this parcel
    """
    groups = sorted(parcel_data.keys())
    group_values = [parcel_data[g] for g in groups]
    
    # Remove NaN/inf
    group_values_clean = [vals[np.isfinite(vals)] for vals in group_values]
    
    # Filter out groups with no valid data
    valid_groups = [(g, vals) for g, vals in zip(groups, group_values_clean) if len(vals) > 0]
    
    if len(valid_groups) < 2:
        return {
            'primary_statistic': np.nan,
            'primary_pvalue': np.nan,
            'n_groups': len(valid_groups)
        }
    
    groups_clean, values_clean = zip(*valid_groups)
    
    # Primary test
    if test_type == 'kruskal_wallis':
        stat, pval = kruskal(*values_clean)
    elif test_type == 'anova':
        stat, pval = stats.f_oneway(*values_clean)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Post-hoc pairwise tests
    posthoc_results = []
    for i in range(len(groups_clean)):
        for j in range(i + 1, len(groups_clean)):
            g1, g2 = groups_clean[i], groups_clean[j]
            v1, v2 = values_clean[i], values_clean[j]
            
            if posthoc_test == 'mannwhitneyu':
                posthoc_stat, posthoc_pval = mannwhitneyu(v1, v2, alternative='two-sided')
            elif posthoc_test == 'ttest':
                posthoc_stat, posthoc_pval = ttest_ind(v1, v2, equal_var=False)
            else:
                raise ValueError(f"Unknown posthoc test: {posthoc_test}")
            
            posthoc_results.append({
                'group1': g1,
                'group2': g2,
                'statistic': posthoc_stat,
                'pvalue': posthoc_pval
            })
    
    results = {
        'primary_statistic': stat,
        'primary_pvalue': pval,
        'n_groups': len(valid_groups),
        'posthoc_tests': posthoc_results
    }
    
    return results


def perform_parcel_level_comparisons(
    unified_df: pd.DataFrame,
    test_type: str = 'kruskal_wallis',
    posthoc_test: str = 'mannwhitneyu',
    correction_methods: List[str] = ['fdr_bh', 'bonferroni'],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform statistical comparisons for each parcel.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    test_type : str
        Primary test type ('kruskal_wallis' or 'anova')
    posthoc_test : str
        Post-hoc test type ('mannwhitneyu' or 'ttest')
    correction_methods : list
        List of correction methods to apply
    alpha : float
        Significance level
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with results for each parcel
    """
    n_parcels = unified_df['parcel_id'].nunique()
    groups = sorted(unified_df['group'].unique())
    
    results_list = []
    primary_pvalues = []
    
    logger.info(f"Performing parcel-level comparisons for {n_parcels} parcels...")
    
    for parcel_id in range(n_parcels):
        # Get data for this parcel across all groups
        parcel_df = unified_df[unified_df['parcel_id'] == parcel_id]
        
        parcel_data = {}
        for group in groups:
            group_data = parcel_df[parcel_df['group'] == group]['sdi_value'].values
            parcel_data[group] = group_data
        
        # Perform comparison
        parcel_results = compare_parcel_across_groups(
            parcel_data,
            test_type=test_type,
            posthoc_test=posthoc_test
        )
        
        # Extract group means
        group_means = {}
        for group in groups:
            group_data = parcel_df[parcel_df['group'] == group]['sdi_value'].values
            group_data_clean = group_data[np.isfinite(group_data)]
            group_means[group] = np.mean(group_data_clean) if len(group_data_clean) > 0 else np.nan
        
        # Build result dictionary
        result_dict = {
            'parcel_id': parcel_id,
            'primary_statistic': parcel_results['primary_statistic'],
            'primary_pvalue': parcel_results['primary_pvalue'],
            'n_groups': parcel_results['n_groups']
        }
        
        # Add group means
        for group in groups:
            result_dict[f'mean_{group}'] = group_means[group]
        
        # Add post-hoc results (store first comparison for now)
        if parcel_results['posthoc_tests']:
            first_posthoc = parcel_results['posthoc_tests'][0]
            result_dict['posthoc_group1'] = first_posthoc['group1']
            result_dict['posthoc_group2'] = first_posthoc['group2']
            result_dict['posthoc_statistic'] = first_posthoc['statistic']
            result_dict['posthoc_pvalue'] = first_posthoc['pvalue']
        else:
            result_dict['posthoc_group1'] = None
            result_dict['posthoc_group2'] = None
            result_dict['posthoc_statistic'] = np.nan
            result_dict['posthoc_pvalue'] = np.nan
        
        results_list.append(result_dict)
        primary_pvalues.append(parcel_results['primary_pvalue'])
        
        if (parcel_id + 1) % 50 == 0:
            logger.info(f"Processed {parcel_id + 1}/{n_parcels} parcels...")
    
    results_df = pd.DataFrame(results_list)
    
    # Apply multiple comparisons correction
    logger.info("Applying multiple comparisons correction...")
    primary_pvalues_array = np.array(primary_pvalues)
    valid_mask = ~np.isnan(primary_pvalues_array)
    
    for correction_method in correction_methods:
        corrected_pvalues = np.full(len(primary_pvalues_array), np.nan)
        if np.sum(valid_mask) > 0:
            _, p_corrected, _, _ = multipletests(
                primary_pvalues_array[valid_mask],
                method=correction_method,
                alpha=alpha
            )
            corrected_pvalues[valid_mask] = p_corrected
        
        results_df[f'pvalue_{correction_method}'] = corrected_pvalues
        results_df[f'significant_{correction_method}'] = corrected_pvalues < alpha
    
    logger.info(f"Completed parcel-level comparisons for {n_parcels} parcels")
    
    return results_df


def create_significance_map(
    parcel_results: pd.DataFrame,
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05
) -> np.ndarray:
    """
    Create binary significance map for parcels.
    
    Parameters:
    -----------
    parcel_results : pd.DataFrame
        Results from perform_parcel_level_comparisons
    correction_method : str
        Correction method to use
    alpha : float
        Significance level
    
    Returns:
    --------
    significance_map : np.ndarray
        Binary array (1 = significant, 0 = not significant)
    """
    col_name = f'significant_{correction_method}'
    if col_name not in parcel_results.columns:
        raise ValueError(f"Column {col_name} not found in results")
    
    # Sort by parcel_id to ensure correct order
    parcel_results_sorted = parcel_results.sort_values('parcel_id')
    significance_map = parcel_results_sorted[col_name].fillna(False).astype(int).values
    
    return significance_map


def create_continuous_significance_map(
    parcel_results: pd.DataFrame,
    correction_method: str = 'fdr_bh'
) -> np.ndarray:
    """
    Create continuous significance map using -log10(p-values).
    
    Parameters:
    -----------
    parcel_results : pd.DataFrame
        Results from perform_parcel_level_comparisons
    correction_method : str
        Correction method to use
    
    Returns:
    --------
    significance_map : np.ndarray
        Array of -log10(p-values)
    """
    col_name = f'pvalue_{correction_method}'
    if col_name not in parcel_results.columns:
        raise ValueError(f"Column {col_name} not found in results")
    
    # Sort by parcel_id
    parcel_results_sorted = parcel_results.sort_values('parcel_id')
    pvalues = parcel_results_sorted[col_name].values
    
    # Compute -log10(p), handling NaN and zero values
    neg_log10_p = np.full_like(pvalues, np.nan, dtype=float)
    valid_mask = (pvalues > 0) & ~np.isnan(pvalues)
    neg_log10_p[valid_mask] = -np.log10(pvalues[valid_mask])
    
    return neg_log10_p


def export_parcel_comparisons(
    parcel_results: pd.DataFrame,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export parcel comparison results to CSV.
    
    Parameters:
    -----------
    parcel_results : pd.DataFrame
        Results from perform_parcel_level_comparisons
    output_dir : str
        Output directory
    prefix : str
        File prefix
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping result type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Full results
    results_file = output_path / f"{prefix}_parcel_comparisons.csv"
    parcel_results.to_csv(results_file, index=False)
    file_paths['full_results'] = str(results_file)
    logger.info(f"Exported parcel comparisons to {results_file}")
    
    # Significant parcels only (FDR)
    if 'significant_fdr_bh' in parcel_results.columns:
        sig_parcels = parcel_results[parcel_results['significant_fdr_bh'] == True]
        sig_file = output_path / f"{prefix}_significant_parcels_fdr.csv"
        sig_parcels.to_csv(sig_file, index=False)
        file_paths['significant_fdr'] = str(sig_file)
        logger.info(f"Found {len(sig_parcels)} significant parcels (FDR-corrected)")
    
    # Significant parcels only (Bonferroni)
    if 'significant_bonferroni' in parcel_results.columns:
        sig_parcels = parcel_results[parcel_results['significant_bonferroni'] == True]
        sig_file = output_path / f"{prefix}_significant_parcels_bonferroni.csv"
        sig_parcels.to_csv(sig_file, index=False)
        file_paths['significant_bonferroni'] = str(sig_file)
        logger.info(f"Found {len(sig_parcels)} significant parcels (Bonferroni-corrected)")
    
    return file_paths


if __name__ == "__main__":
    # Test
    from load_sdi_data import load_all_sdi_data
    from utils.load_config import load_config
    
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    unified_df, _ = load_all_sdi_data(config)
    
    results = perform_parcel_level_comparisons(unified_df, test_type='kruskal_wallis')
    print(f"\nParcel comparison results shape: {results.shape}")
    print(f"\nSignificant parcels (FDR): {results['significant_fdr_bh'].sum()}")
    print(f"Significant parcels (Bonferroni): {results['significant_bonferroni'].sum()}")



