"""
Parcel-level statistical comparisons for SDI data.

This module performs group comparisons for each parcel with multiple comparisons correction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging
from scipy.stats import kruskal, f_oneway, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def compare_parcel_across_groups(
    parcel_data: Dict[str, np.ndarray],
    test_type: str = 'kruskal_wallis',
    posthoc_test: str = 'mannwhitneyu'
) -> Dict:
    """
    Compare a single parcel across groups.
    
    Parameters:
    -----------
    parcel_data : dict
        Dictionary mapping group names to data arrays
    test_type : str
        Primary test: 'kruskal_wallis' or 'anova' (default: 'kruskal_wallis')
    posthoc_test : str
        Post-hoc test: 'mannwhitneyu' or 'ttest' (default: 'mannwhitneyu')
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    groups = sorted(parcel_data.keys())
    group_data_list = []
    
    for group in groups:
        data = parcel_data[group]
        data_clean = data[np.isfinite(data)]
        if len(data_clean) > 0:
            group_data_list.append(data_clean)
        else:
            group_data_list.append(np.array([]))
    
    # Filter out empty groups
    valid_groups = [(g, data) for g, data in zip(groups, group_data_list) if len(data) > 0]
    
    if len(valid_groups) < 2:
        return {
            'test': test_type,
            'statistic': None,
            'pvalue': None,
            'posthoc_results': []
        }
    
    groups = [g for g, _ in valid_groups]
    group_data_list = [data for _, data in valid_groups]
    
    # Primary test
    if test_type == 'kruskal_wallis':
        stat, pval = kruskal(*group_data_list)
    elif test_type == 'anova':
        stat, pval = f_oneway(*group_data_list)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Post-hoc tests
    posthoc_results = []
    n_groups = len(groups)
    
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            group1, group2 = groups[i], groups[j]
            data1, data2 = group_data_list[i], group_data_list[j]
            
            if posthoc_test == 'mannwhitneyu':
                stat_post, pval_post = mannwhitneyu(data1, data2, alternative='two-sided')
            elif posthoc_test == 'ttest':
                stat_post, pval_post = ttest_ind(data1, data2, equal_var=False)
            else:
                raise ValueError(f"Unknown post-hoc test type: {posthoc_test}")
            
            posthoc_results.append({
                'group1': group1,
                'group2': group2,
                'statistic': stat_post,
                'pvalue': pval_post
            })
    
    return {
        'test': test_type,
        'statistic': stat,
        'pvalue': pval,
        'posthoc_results': posthoc_results
    }


def perform_parcel_level_comparisons(
    unified_df: pd.DataFrame,
    use_parametric: bool = False,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform group comparisons for each parcel.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    use_parametric : bool
        Whether to use parametric tests (default: False)
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    parcel_results_df : pd.DataFrame
        DataFrame with parcel-level comparison results
    """
    parcels = sorted(unified_df['parcel_id'].unique())
    groups = sorted(unified_df['group'].unique())
    
    test_type = 'anova' if use_parametric else 'kruskal_wallis'
    posthoc_test = 'ttest' if use_parametric else 'mannwhitneyu'
    
    results_list = []
    
    logger.info(f"Performing parcel-level comparisons for {len(parcels)} parcels...")
    
    for parcel_id in parcels:
        parcel_df = unified_df[unified_df['parcel_id'] == parcel_id]
        
        # Organize data by group
        parcel_data = {}
        for group in groups:
            group_data = parcel_df[parcel_df['group'] == group]['sdi_value'].values
            parcel_data[group] = group_data
        
        # Perform comparison
        result = compare_parcel_across_groups(parcel_data, test_type=test_type, posthoc_test=posthoc_test)
        
        results_list.append({
            'parcel_id': parcel_id,
            'test': result['test'],
            'statistic': result['statistic'],
            'pvalue': result['pvalue']
        })
        
        if (parcel_id % 50 == 0) and parcel_id > 0:
            logger.info(f"  Processed {parcel_id}/{len(parcels)} parcels...")
    
    parcel_results_df = pd.DataFrame(results_list)
    logger.info(f"Completed parcel-level comparisons")
    
    return parcel_results_df


def apply_multiple_comparisons_correction(
    parcel_results_df: pd.DataFrame,
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Apply multiple comparisons correction to parcel-level p-values.
    
    Parameters:
    -----------
    parcel_results_df : pd.DataFrame
        DataFrame with parcel-level p-values
    correction_method : str
        Correction method: 'fdr_bh' or 'bonferroni' (default: 'fdr_bh')
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    corrected_df : pd.DataFrame
        DataFrame with corrected p-values
    """
    # Filter out NaN p-values
    valid_results = parcel_results_df[parcel_results_df['pvalue'].notna()].copy()
    
    if len(valid_results) == 0:
        parcel_results_df['pvalue_corrected'] = np.nan
        parcel_results_df['significant'] = False
        return parcel_results_df
    
    p_values = valid_results['pvalue'].values
    
    # Apply correction
    if correction_method == 'fdr_bh':
        _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    elif correction_method == 'bonferroni':
        _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    else:
        raise ValueError(f"Unknown correction method: {correction_method}")
    
    # Add corrected p-values
    valid_results['pvalue_corrected'] = p_corrected
    valid_results['significant'] = p_corrected < alpha
    
    # Merge back with original dataframe
    parcel_results_df = parcel_results_df.merge(
        valid_results[['parcel_id', 'pvalue_corrected', 'significant']],
        on='parcel_id',
        how='left'
    )
    
    return parcel_results_df


def create_significance_map(
    parcel_results_df: pd.DataFrame,
    correction_method: str = 'fdr_bh',
    n_parcels: int = 430
) -> np.ndarray:
    """
    Create binary significance map from parcel results.
    
    Parameters:
    -----------
    parcel_results_df : pd.DataFrame
        DataFrame with parcel-level results (must have 'significant' column)
    correction_method : str
        Correction method used (for logging)
    n_parcels : int
        Total number of parcels (default: 430)
    
    Returns:
    --------
    significance_map : np.ndarray
        Binary array (1 = significant, 0 = not significant)
    """
    significance_map = np.zeros(n_parcels, dtype=int)
    
    if 'significant' in parcel_results_df.columns:
        for _, row in parcel_results_df.iterrows():
            parcel_id = int(row['parcel_id'])
            if 1 <= parcel_id <= n_parcels:
                if row.get('significant', False):
                    significance_map[parcel_id - 1] = 1  # Convert to 0-indexed
    
    n_significant = np.sum(significance_map)
    logger.info(f"Significance map ({correction_method}): {n_significant} significant parcels")
    
    return significance_map


def create_continuous_significance_map(
    parcel_results_df: pd.DataFrame,
    n_parcels: int = 430,
    use_corrected: bool = True
) -> np.ndarray:
    """
    Create continuous significance map (-log10(p)) from parcel results.
    
    Parameters:
    -----------
    parcel_results_df : pd.DataFrame
        DataFrame with parcel-level results
    n_parcels : int
        Total number of parcels (default: 430)
    use_corrected : bool
        Use corrected p-values if available (default: True)
    
    Returns:
    --------
    significance_map : np.ndarray
        Continuous array with -log10(p) values
    """
    significance_map = np.zeros(n_parcels)
    
    p_col = 'pvalue_corrected' if (use_corrected and 'pvalue_corrected' in parcel_results_df.columns) else 'pvalue'
    
    for _, row in parcel_results_df.iterrows():
        parcel_id = int(row['parcel_id'])
        if 1 <= parcel_id <= n_parcels:
            pval = row.get(p_col)
            if pd.notna(pval) and pval > 0:
                significance_map[parcel_id - 1] = -np.log10(pval)  # Convert to 0-indexed
    
    return significance_map


def export_parcel_comparisons(
    parcel_results_df: pd.DataFrame,
    output_dir: str,
    prefix: str = "sdi",
    apply_corrections: bool = True,
    alpha: float = 0.05
) -> Dict[str, str]:
    """
    Export parcel comparison results to CSV files.
    
    Parameters:
    -----------
    parcel_results_df : pd.DataFrame
        DataFrame with parcel-level results
    output_dir : str
        Output directory
    prefix : str
        File prefix
    apply_corrections : bool
        Apply multiple comparisons corrections (default: True)
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping result type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Full results
    full_file = output_path / f"{prefix}_parcel_comparisons.csv"
    parcel_results_df.to_csv(full_file, index=False)
    file_paths['full'] = str(full_file)
    logger.info(f"Saved parcel comparisons to {full_file}")
    
    if apply_corrections:
        # FDR correction
        fdr_df = apply_multiple_comparisons_correction(
            parcel_results_df.copy(),
            correction_method='fdr_bh',
            alpha=alpha
        )
        fdr_file = output_path / f"{prefix}_significant_parcels_fdr.csv"
        fdr_df.to_csv(fdr_file, index=False)
        file_paths['fdr'] = str(fdr_file)
        logger.info(f"Saved FDR-corrected results to {fdr_file}")
        
        # Bonferroni correction
        bonf_df = apply_multiple_comparisons_correction(
            parcel_results_df.copy(),
            correction_method='bonferroni',
            alpha=alpha
        )
        bonf_file = output_path / f"{prefix}_significant_parcels_bonferroni.csv"
        bonf_df.to_csv(bonf_file, index=False)
        file_paths['bonferroni'] = str(bonf_file)
        logger.info(f"Saved Bonferroni-corrected results to {bonf_file}")
        
        # Significant parcels only (FDR)
        significant_fdr = fdr_df[fdr_df['significant'] == True].copy()
        if len(significant_fdr) > 0:
            sig_fdr_file = output_path / f"{prefix}_significant_parcels_fdr_only.csv"
            significant_fdr.to_csv(sig_fdr_file, index=False)
            file_paths['significant_fdr'] = str(sig_fdr_file)
        
        # Significant parcels only (Bonferroni)
        significant_bonf = bonf_df[bonf_df['significant'] == True].copy()
        if len(significant_bonf) > 0:
            sig_bonf_file = output_path / f"{prefix}_significant_parcels_bonferroni_only.csv"
            significant_bonf.to_csv(sig_bonf_file, index=False)
            file_paths['significant_bonferroni'] = str(sig_bonf_file)
    
    return file_paths




