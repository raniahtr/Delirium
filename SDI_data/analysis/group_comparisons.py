"""
Group-level statistical comparisons for SDI data.

This module performs Kruskal-Wallis, ANOVA, post-hoc tests, and computes
effect sizes with bootstrap confidence intervals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats
from scipy.stats import kruskal, f_oneway, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def compute_effect_size_cohens_d(group1_data: np.ndarray, group2_data: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Parameters:
    -----------
    group1_data : np.ndarray
        Data for group 1
    group2_data : np.ndarray
        Data for group 2
    
    Returns:
    --------
    cohens_d : float
        Cohen's d effect size
    """
    n1, n2 = len(group1_data), len(group2_data)
    var1, var2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
    return cohens_d


def compute_effect_size_cliffs_delta(group1_data: np.ndarray, group2_data: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size (non-parametric).
    
    Parameters:
    -----------
    group1_data : np.ndarray
        Data for group 1
    group2_data : np.ndarray
        Data for group 2
    
    Returns:
    --------
    cliffs_delta : float
        Cliff's delta effect size (-1 to 1)
    """
    n1, n2 = len(group1_data), len(group2_data)
    
    # Count dominances
    dominance = 0
    for x in group1_data:
        for y in group2_data:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    
    cliffs_delta = dominance / (n1 * n2)
    return cliffs_delta


def bootstrap_confidence_interval(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for group difference.
    
    Parameters:
    -----------
    group1_data : np.ndarray
        Data for group 1
    group2_data : np.ndarray
        Data for group 2
    statistic_func : callable
        Function to compute statistic (default: mean)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (default: 0.95)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    mean_diff : float
        Mean difference
    ci_lower : float
        Lower confidence interval bound
    ci_upper : float
        Upper confidence interval bound
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n1, n2 = len(group1_data), len(group2_data)
    differences = []
    
    for _ in range(n_bootstrap):
        # Bootstrap samples
        boot1 = np.random.choice(group1_data, size=n1, replace=True)
        boot2 = np.random.choice(group2_data, size=n2, replace=True)
        
        # Compute difference
        diff = statistic_func(boot1) - statistic_func(boot2)
        differences.append(diff)
    
    differences = np.array(differences)
    mean_diff = np.mean(differences)
    
    # Percentile-based CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(differences, 100 * alpha / 2)
    ci_upper = np.percentile(differences, 100 * (1 - alpha / 2))
    
    return mean_diff, ci_lower, ci_upper


def perform_kruskal_wallis_test(
    unified_df: pd.DataFrame
) -> Dict:
    """
    Perform Kruskal-Wallis test (non-parametric one-way ANOVA).
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    results : dict
        Test results
    """
    groups = sorted(unified_df['group'].unique())
    group_data = [unified_df[unified_df['group'] == g]['sdi_value'].values 
                  for g in groups]
    
    # Remove NaN/inf
    group_data_clean = [data[np.isfinite(data)] for data in group_data]
    
    if len(group_data_clean) < 2:
        raise ValueError("Need at least 2 groups for Kruskal-Wallis test")
    
    h_stat, p_value = kruskal(*group_data_clean)
    
    results = {
        'test': 'kruskal_wallis',
        'statistic': h_stat,
        'pvalue': p_value,
        'significant': p_value < 0.05,
        'groups': groups,
        'n_per_group': [len(data) for data in group_data_clean]
    }
    
    return results


def perform_anova_test(
    unified_df: pd.DataFrame
) -> Dict:
    """
    Perform one-way ANOVA test (parametric).
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    results : dict
        Test results
    """
    groups = sorted(unified_df['group'].unique())
    group_data = [unified_df[unified_df['group'] == g]['sdi_value'].values 
                  for g in groups]
    
    # Remove NaN/inf
    group_data_clean = [data[np.isfinite(data)] for data in group_data]
    
    if len(group_data_clean) < 2:
        raise ValueError("Need at least 2 groups for ANOVA test")
    
    f_stat, p_value = f_oneway(*group_data_clean)
    
    results = {
        'test': 'anova',
        'statistic': f_stat,
        'pvalue': p_value,
        'significant': p_value < 0.05,
        'groups': groups,
        'n_per_group': [len(data) for data in group_data_clean]
    }
    
    return results


def perform_posthoc_tests(
    unified_df: pd.DataFrame,
    test_type: str = 'mannwhitneyu',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform pairwise post-hoc tests with multiple comparisons correction.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    test_type : str
        'mannwhitneyu' (non-parametric) or 'ttest' (parametric)
    correction_method : str
        Multiple comparisons correction method (default: 'fdr_bh')
    alpha : float
        Significance level
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with post-hoc test results
    """
    groups = sorted(unified_df['group'].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for post-hoc tests")
    
    results_list = []
    p_values = []
    
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            group1, group2 = groups[i], groups[j]
            data1 = unified_df[unified_df['group'] == group1]['sdi_value'].values
            data2 = unified_df[unified_df['group'] == group2]['sdi_value'].values
            
            # Remove NaN/inf
            data1_clean = data1[np.isfinite(data1)]
            data2_clean = data2[np.isfinite(data2)]
            
            if len(data1_clean) == 0 or len(data2_clean) == 0:
                continue
            
            # Perform test
            if test_type == 'mannwhitneyu':
                stat, pval = mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
                test_name = 'Mann-Whitney U'
            elif test_type == 'ttest':
                stat, pval = ttest_ind(data1_clean, data2_clean, equal_var=False)
                test_name = "Welch's t-test"
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Compute effect sizes
            if test_type == 'ttest':
                effect_size = compute_effect_size_cohens_d(data1_clean, data2_clean)
                effect_size_name = "Cohen's d"
            else:
                effect_size = compute_effect_size_cliffs_delta(data1_clean, data2_clean)
                effect_size_name = "Cliff's delta"
            
            # Bootstrap CI for mean difference
            mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                data1_clean, data2_clean, random_seed=42
            )
            
            results_list.append({
                'group1': group1,
                'group2': group2,
                'test': test_name,
                'statistic': stat,
                'pvalue_uncorrected': pval,
                'mean1': np.mean(data1_clean),
                'mean2': np.mean(data2_clean),
                'mean_diff': mean_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'effect_size': effect_size,
                'effect_size_name': effect_size_name,
                'n1': len(data1_clean),
                'n2': len(data2_clean)
            })
            p_values.append(pval)
    
    # Multiple comparisons correction
    if len(p_values) > 0:
        _, p_corrected, _, _ = multipletests(p_values, method=correction_method, alpha=alpha)
        
        for idx, result in enumerate(results_list):
            result['pvalue_corrected'] = p_corrected[idx]
            result['significant_uncorrected'] = result['pvalue_uncorrected'] < alpha
            result['significant_corrected'] = p_corrected[idx] < alpha
            result['correction_method'] = correction_method
    
    results_df = pd.DataFrame(results_list)
    return results_df


def perform_group_comparisons(
    unified_df: pd.DataFrame,
    use_parametric: bool = False,
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Dict:
    """
    Perform comprehensive group-level comparisons.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    use_parametric : bool
        Whether to use parametric tests (default: False, use non-parametric)
    correction_method : str
        Multiple comparisons correction method
    alpha : float
        Significance level
    
    Returns:
    --------
    results : dict
        Dictionary with all test results
    """
    results = {}
    
    # Primary test
    if use_parametric:
        primary_test = perform_anova_test(unified_df)
        posthoc_test = 'ttest'
    else:
        primary_test = perform_kruskal_wallis_test(unified_df)
        posthoc_test = 'mannwhitneyu'
    
    results['primary_test'] = primary_test
    
    # Post-hoc tests (only if primary test is significant)
    if primary_test['significant']:
        posthoc_results = perform_posthoc_tests(
            unified_df,
            test_type=posthoc_test,
            correction_method=correction_method,
            alpha=alpha
        )
        results['posthoc_tests'] = posthoc_results
    else:
        logger.info("Primary test not significant, skipping post-hoc tests")
        results['posthoc_tests'] = pd.DataFrame()
    
    return results


def export_group_comparisons(
    comparison_results: Dict,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export group comparison results to CSV.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from perform_group_comparisons
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
    
    # Primary test results
    primary_df = pd.DataFrame([comparison_results['primary_test']])
    primary_file = output_path / f"{prefix}_primary_test.csv"
    primary_df.to_csv(primary_file, index=False)
    file_paths['primary_test'] = str(primary_file)
    
    # Post-hoc test results
    if len(comparison_results['posthoc_tests']) > 0:
        posthoc_file = output_path / f"{prefix}_posthoc_tests.csv"
        comparison_results['posthoc_tests'].to_csv(posthoc_file, index=False)
        file_paths['posthoc_tests'] = str(posthoc_file)
    else:
        file_paths['posthoc_tests'] = None
    
    return file_paths


if __name__ == "__main__":
    # Test
    from load_sdi_data import load_all_sdi_data
    from utils.load_config import load_config
    
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    unified_df, _ = load_all_sdi_data(config)
    
    results = perform_group_comparisons(unified_df, use_parametric=False)
    print("\nPrimary test results:")
    print(results['primary_test'])
    print("\nPost-hoc test results:")
    print(results['posthoc_tests'])



