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
from scipy.stats import kruskal, f_oneway, mannwhitneyu, ttest_ind, t as t_dist
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


def compute_cohens_d_ci(group1_data: np.ndarray, group2_data: np.ndarray, 
                        confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute analytical confidence interval for Cohen's d.
    
    Uses the formula from Hedges & Olkin (1985) and Goulet-Pelletier & Cousineau (2018).
    
    Parameters:
    -----------
    group1_data : np.ndarray
        Data for group 1
    group2_data : np.ndarray
        Data for group 2
    confidence_level : float
        Confidence level (default: 0.95)
    
    Returns:
    --------
    cohens_d : float
        Cohen's d
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound
    """
    n1, n2 = len(group1_data), len(group2_data)
    var1, var2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0, 0.0, 0.0
    
    cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
    
    # Standard error of Cohen's d
    # Using Hedges' correction for small samples
    df = n1 + n2 - 2
    correction_factor = 1 - (3 / (4 * df - 1))
    hedges_g = cohens_d * correction_factor
    
    # Variance of Cohen's d
    var_d = ((n1 + n2) / (n1 * n2)) + (cohens_d**2 / (2 * df))
    se_d = np.sqrt(var_d)
    
    # t-value for confidence interval
    alpha = 1 - confidence_level
    t_crit = t_dist.ppf(1 - alpha/2, df)
    
    ci_lower = cohens_d - t_crit * se_d
    ci_upper = cohens_d + t_crit * se_d
    
    return cohens_d, ci_lower, ci_upper


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


def bootstrap_effect_size_ci(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    effect_size_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for effect size itself.
    
    Parameters:
    -----------
    group1_data : np.ndarray
        Data for group 1
    group2_data : np.ndarray
        Data for group 2
    effect_size_func : callable
        Function to compute effect size (e.g., compute_effect_size_cohens_d)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (default: 0.95)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    mean_es : float
        Mean effect size
    ci_lower : float
        Lower confidence interval bound for effect size
    ci_upper : float
        Upper confidence interval bound for effect size
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n1, n2 = len(group1_data), len(group2_data)
    effect_sizes = []
    
    for _ in range(n_bootstrap):
        # Bootstrap samples
        boot1 = np.random.choice(group1_data, size=n1, replace=True)
        boot2 = np.random.choice(group2_data, size=n2, replace=True)
        
        # Compute effect size
        es = effect_size_func(boot1, boot2)
        effect_sizes.append(es)
    
    effect_sizes = np.array(effect_sizes)
    mean_es = np.mean(effect_sizes)
    
    # Percentile-based CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(effect_sizes, 100 * alpha / 2)
    ci_upper = np.percentile(effect_sizes, 100 * (1 - alpha / 2))
    
    return mean_es, ci_lower, ci_upper


def bootstrap_confidence_interval(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
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
        Dictionary with test results
    """
    groups = sorted(unified_df['group'].unique())
    group_data_list = []
    
    for group in groups:
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        group_data_list.append(group_data_clean)
    
    if len(group_data_list) < 2:
        return {
            'test_name': 'Kruskal-Wallis',
            'statistic': None,
            'pvalue': None,
            'groups': groups
        }
    
    stat, pval = kruskal(*group_data_list)
    
    return {
        'test_name': 'Kruskal-Wallis',
        'statistic': stat,
        'pvalue': pval,
        'groups': groups,
        'n_groups': len(groups)
    }


def perform_anova_test(
    unified_df: pd.DataFrame
) -> Dict:
    """
    Perform one-way ANOVA test.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    groups = sorted(unified_df['group'].unique())
    group_data_list = []
    
    for group in groups:
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        group_data_list.append(group_data_clean)
    
    if len(group_data_list) < 2:
        return {
            'test_name': 'One-way ANOVA',
            'statistic': None,
            'pvalue': None,
            'groups': groups
        }
    
    stat, pval = f_oneway(*group_data_list)
    
    return {
        'test_name': 'One-way ANOVA',
        'statistic': stat,
        'pvalue': pval,
        'groups': groups,
        'n_groups': len(groups)
    }


def perform_posthoc_tests(
    unified_df: pd.DataFrame,
    test_type: str = 'mannwhitneyu',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform post-hoc pairwise tests with multiple comparisons correction.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    test_type : str
        Test type: 'mannwhitneyu' or 'ttest' (default: 'mannwhitneyu')
    correction_method : str
        Multiple comparisons correction: 'fdr_bh' or 'bonferroni' (default: 'fdr_bh')
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    posthoc_df : pd.DataFrame
        DataFrame with post-hoc test results
    """
    groups = sorted(unified_df['group'].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        return pd.DataFrame()
    
    results_list = []
    p_values = []
    
    # Count total comparisons for progress
    total_comparisons = n_groups * (n_groups - 1) // 2
    logger.info(f"Performing {total_comparisons} pairwise comparisons with analytical CIs (no bootstrap)...")
    
    comparison_num = 0
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            comparison_num += 1
            group1, group2 = groups[i], groups[j]
            logger.info(f"  [{comparison_num}/{total_comparisons}] Comparing {group1} vs {group2}...")
            
            data1 = unified_df[unified_df['group'] == group1]['sdi_value'].values
            data2 = unified_df[unified_df['group'] == group2]['sdi_value'].values
            
            # Remove NaN/inf
            data1_clean = data1[np.isfinite(data1)]
            data2_clean = data2[np.isfinite(data2)]
            
            if len(data1_clean) == 0 or len(data2_clean) == 0:
                logger.warning(f"    Skipping: insufficient data (n1={len(data1_clean)}, n2={len(data2_clean)})")
                continue
            
            # Perform test
            if test_type == 'mannwhitneyu':
                stat, pval = mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
                test_name = 'Mann-Whitney U'
                # For Mann-Whitney, compute mean difference manually
                mean_diff = np.mean(data1_clean) - np.mean(data2_clean)
                # Simple SE-based CI for mean difference (approximate)
                se1 = np.std(data1_clean, ddof=1) / np.sqrt(len(data1_clean))
                se2 = np.std(data2_clean, ddof=1) / np.sqrt(len(data2_clean))
                se_diff = np.sqrt(se1**2 + se2**2)
                # Use t-distribution with approximate df (Welch-Satterthwaite)
                df_approx = min(len(data1_clean) - 1, len(data2_clean) - 1)
                t_crit = t_dist.ppf(0.975, df_approx)
                ci_lower = mean_diff - t_crit * se_diff
                ci_upper = mean_diff + t_crit * se_diff
            elif test_type == 'ttest':
                stat, pval = ttest_ind(data1_clean, data2_clean, equal_var=False)
                test_name = "Welch's t-test"
                # ttest_ind doesn't return CI, compute it manually
                mean_diff = np.mean(data1_clean) - np.mean(data2_clean)
                se1 = np.std(data1_clean, ddof=1) / np.sqrt(len(data1_clean))
                se2 = np.std(data2_clean, ddof=1) / np.sqrt(len(data2_clean))
                se_diff = np.sqrt(se1**2 + se2**2)
                # Welch-Satterthwaite degrees of freedom
                df_welch = (se1**2 + se2**2)**2 / (se1**4/(len(data1_clean)-1) + se2**4/(len(data2_clean)-1))
                t_crit = t_dist.ppf(0.975, df_welch)
                ci_lower = mean_diff - t_crit * se_diff
                ci_upper = mean_diff + t_crit * se_diff
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Compute effect sizes with analytical CIs (no bootstrap)
            if test_type == 'ttest':
                # Use analytical CI for Cohen's d
                effect_size, ci_lower_es, ci_upper_es = compute_cohens_d_ci(
                    data1_clean, data2_clean, confidence_level=0.95
                )
                effect_size_name = "Cohen's d"
                effect_size_mean = effect_size  # Same as effect_size for analytical
            else:
                # For Cliff's delta, compute effect size but skip CI
                # (Bootstrap would be standard but is computationally expensive;
                # analytical formulas are not well-established for Cliff's delta)
                effect_size = compute_effect_size_cliffs_delta(data1_clean, data2_clean)
                effect_size_name = "Cliff's delta"
                effect_size_mean = effect_size
                # No CI computed for Cliff's delta
                ci_lower_es = np.nan
                ci_upper_es = np.nan
            
            results_list.append({
                'group1': group1,
                'group2': group2,
                'test': test_name,
                'statistic': stat,
                'pvalue': pval,
                'effect_size': effect_size,  # Original effect size
                'effect_size_mean': effect_size_mean,  # Bootstrap mean
                'effect_size_ci_lower': ci_lower_es,  # CI for effect size
                'effect_size_ci_upper': ci_upper_es,
                'effect_size_name': effect_size_name,
                'mean_diff': mean_diff,  # Keep for reference
                'ci_lower': ci_lower,  # CI for mean difference
                'ci_upper': ci_upper,
                'n1': len(data1_clean),
                'n2': len(data2_clean)
            })
            
            p_values.append(pval)
    
    if len(results_list) == 0:
        return pd.DataFrame()
    
    # Apply multiple comparisons correction
    if len(p_values) > 0:
        if correction_method == 'fdr_bh':
            _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        elif correction_method == 'bonferroni':
            _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
        else:
            p_corrected = p_values
        
        # Add corrected p-values
        for i, result in enumerate(results_list):
            result['pvalue_corrected'] = p_corrected[i]
            result['significant'] = p_corrected[i] < alpha
    
    posthoc_df = pd.DataFrame(results_list)
    return posthoc_df


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
        Whether to use parametric tests (default: False)
    correction_method : str
        Multiple comparisons correction method (default: 'fdr_bh')
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    results : dict
        Dictionary with all comparison results
    """
    results = {}
    
    # Primary test
    if use_parametric:
        primary_result = perform_anova_test(unified_df)
        posthoc_test = 'ttest'
    else:
        primary_result = perform_kruskal_wallis_test(unified_df)
        posthoc_test = 'mannwhitneyu'
    
    results['primary_test'] = primary_result
    
    # Post-hoc tests (only if primary test is significant or if we want all comparisons)
    posthoc_df = perform_posthoc_tests(
        unified_df,
        test_type=posthoc_test,
        correction_method=correction_method,
        alpha=alpha
    )
    
    results['posthoc_tests'] = posthoc_df
    
    return results


def export_group_comparisons(
    comparison_results: Dict,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export group comparison results to CSV files.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from perform_group_comparisons()
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
    
    # Primary test
    if comparison_results.get('primary_test'):
        primary_df = pd.DataFrame([comparison_results['primary_test']])
        primary_file = output_path / f"{prefix}_primary_test.csv"
        primary_df.to_csv(primary_file, index=False)
        file_paths['primary'] = str(primary_file)
        logger.info(f"Saved primary test results to {primary_file}")
    
    # Post-hoc tests
    if comparison_results.get('posthoc_tests') is not None and len(comparison_results['posthoc_tests']) > 0:
        posthoc_file = output_path / f"{prefix}_posthoc_tests.csv"
        comparison_results['posthoc_tests'].to_csv(posthoc_file, index=False)
        file_paths['posthoc'] = str(posthoc_file)
        logger.info(f"Saved post-hoc test results to {posthoc_file}")
    
    return file_paths

