"""
Statistical assumption verification for SDI data.

This module performs normality tests, homogeneity of variance tests,
and provides recommendations for appropriate statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
from scipy.stats import shapiro, normaltest, levene, bartlett
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


def test_normality(
    data: np.ndarray,
    group_name: str = "data"
) -> Dict[str, float]:
    """
    Test normality of data using appropriate test.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array to test
    group_name : str
        Name of the group (for logging)
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    results = {
        'group': group_name,
        'n': len(data),
        'test_used': None,
        'statistic': None,
        'pvalue': None,
        'is_normal': None
    }
    
    # Remove NaN and infinite values
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) < 3:
        logger.warning(f"Group {group_name}: insufficient data for normality test")
        return results
    
    # Choose test based on sample size
    if len(data_clean) <= 5000:
        # Shapiro-Wilk test (recommended for n <= 5000)
        stat, pval = shapiro(data_clean)
        results['test_used'] = 'shapiro_wilk'
        results['statistic'] = stat
        results['pvalue'] = pval
    else:
        # D'Agostino-Pearson test (for larger samples)
        stat, pval = normaltest(data_clean)
        results['test_used'] = 'dagostino_pearson'
        results['statistic'] = stat
        results['pvalue'] = pval
    
    # Decision (alpha = 0.05)
    results['is_normal'] = pval > 0.05
    
    return results


def test_homogeneity_of_variance(
    group_data: Dict[str, np.ndarray],
    test_type: str = 'levene'
) -> Dict[str, float]:
    """
    Test homogeneity of variance across groups.
    
    Parameters:
    -----------
    group_data : dict
        Dictionary mapping group names to data arrays
    test_type : str
        Test type: 'levene' (robust) or 'bartlett' (requires normality)
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    group_names = list(group_data.keys())
    all_groups_data = [group_data[g][np.isfinite(group_data[g])] for g in group_names]
    
    # Filter out groups with insufficient data
    valid_groups = [(name, data) for name, data in zip(group_names, all_groups_data) if len(data) >= 2]
    if len(valid_groups) < 2:
        return {
            'test_used': test_type,
            'statistic': None,
            'pvalue': None,
            'is_homogeneous': None,
            'groups_tested': []
        }
    
    group_names = [name for name, _ in valid_groups]
    all_groups_data = [data for _, data in valid_groups]
    
    try:
        if test_type == 'levene':
            stat, pval = levene(*all_groups_data)
        elif test_type == 'bartlett':
            stat, pval = bartlett(*all_groups_data)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            'test_used': test_type,
            'statistic': stat,
            'pvalue': pval,
            'is_homogeneous': pval > 0.05,
            'groups_tested': group_names
        }
    except Exception as e:
        logger.warning(f"{test_type} test failed: {e}")
        return {
            'test_used': test_type,
            'statistic': None,
            'pvalue': None,
            'is_homogeneous': None,
            'groups_tested': group_names
        }


def check_all_assumptions(
    unified_df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict:
    """
    Perform comprehensive assumption checks for all groups.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    assumption_results : dict
        Dictionary with all assumption check results
    """
    results = {
        'normality_tests': [],
        'variance_tests': {},
        'recommendations': {}
    }
    
    # Normality tests for each group
    group_data_dict = {}
    for group in unified_df['group'].unique():
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_dict[group] = group_data
        
        norm_result = test_normality(group_data, group)
        results['normality_tests'].append(norm_result)
    
    # Homogeneity of variance tests
    # Levene's test (robust to non-normality)
    levene_result = test_homogeneity_of_variance(group_data_dict, 'levene')
    results['variance_tests']['levene'] = levene_result
    
    # Bartlett's test (if data is normal)
    try:
        bartlett_result = test_homogeneity_of_variance(group_data_dict, 'bartlett')
        results['variance_tests']['bartlett'] = bartlett_result
    except Exception as e:
        logger.warning(f"Bartlett's test failed: {e}")
    
    # Generate recommendations
    all_normal = all(r['is_normal'] for r in results['normality_tests'] if r['pvalue'] is not None)
    variances_homogeneous = levene_result['is_homogeneous'] if levene_result.get('is_homogeneous') is not None else False
    
    recommendations = {
        'use_parametric': all_normal and variances_homogeneous,
        'primary_test': 'ANOVA' if (all_normal and variances_homogeneous) else 'Kruskal-Wallis',
        'posthoc_test': 'Welch_t_test' if (all_normal and variances_homogeneous) else 'Mann-Whitney_U',
        'effect_size': "Cohen's d" if (all_normal and variances_homogeneous) else "Cliff's delta",
        'all_normal': all_normal,
        'variances_homogeneous': variances_homogeneous
    }
    
    results['recommendations'] = recommendations
    
    return results


def plot_qq_plots(
    unified_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (15, 5)
):
    """
    Create Q-Q plots for each group to assess normality.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_path : str
        Path to save figure
    figsize : tuple
        Figure size (default: (15, 5))
    """
    groups = sorted(unified_df['group'].unique())
    n_groups = len(groups)
    
    fig, axes = plt.subplots(1, n_groups, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, group in enumerate(groups):
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        
        if len(group_data_clean) > 0:
            stats.probplot(group_data_clean, dist="norm", plot=axes[i])
            axes[i].set_title(f"{group}\n(n={len(group_data_clean)})")
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(group)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Q-Q plots to {output_path}")


def plot_variance_comparison(
    unified_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6)
):
    """
    Create box plot comparing variances across groups.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_path : str
        Path to save figure
    figsize : tuple
        Figure size (default: (10, 6))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute variance per subject per group
    variance_data = []
    group_labels = []
    
    for group in sorted(unified_df['group'].unique()):
        group_df = unified_df[unified_df['group'] == group]
        
        # Compute variance per subject
        for subject_id in group_df['subject_id'].unique():
            subject_data = group_df[group_df['subject_id'] == subject_id]['sdi_value'].values
            subject_data_clean = subject_data[np.isfinite(subject_data)]
            
            if len(subject_data_clean) > 1:
                variance = np.var(subject_data_clean, ddof=1)
                variance_data.append(variance)
                group_labels.append(group)
    
    if len(variance_data) > 0:
        variance_df = pd.DataFrame({'group': group_labels, 'variance': variance_data})
        
        # Create box plot
        groups = sorted(variance_df['group'].unique())
        data_to_plot = [variance_df[variance_df['group'] == g]['variance'].values for g in groups]
        
        bp = ax.boxplot(data_to_plot, labels=groups, patch_artist=True)
        
        # Color boxes
        from utils.group import get_group_color
        color_dict = get_group_color()
        # Map standard names to internal names for colors
        name_mapping = {'ICU': 'CR', 'ICU Delirium': 'Delirium', 'Healthy Controls': 'HC'}
        
        for patch, group in zip(bp['boxes'], groups):
            internal_name = name_mapping.get(group, group)
            color = color_dict.get(internal_name, '#808080')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Variance')
        ax.set_title('Variance Comparison Across Groups')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved variance comparison plot to {output_path}")


def export_assumption_checks(
    assumption_results: Dict,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export assumption check results to CSV files.
    
    Parameters:
    -----------
    assumption_results : dict
        Results from check_all_assumptions()
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
    
    # Normality tests
    if assumption_results['normality_tests']:
        norm_df = pd.DataFrame(assumption_results['normality_tests'])
        norm_file = output_path / f"{prefix}_normality_tests.csv"
        norm_df.to_csv(norm_file, index=False)
        file_paths['normality'] = str(norm_file)
        logger.info(f"Saved normality tests to {norm_file}")
    
    # Variance tests
    variance_results = []
    for test_name, test_result in assumption_results['variance_tests'].items():
        if test_result:
            result_dict = {
                'test': test_name,
                'statistic': test_result.get('statistic'),
                'pvalue': test_result.get('pvalue'),
                'is_homogeneous': test_result.get('is_homogeneous'),
                'groups_tested': ', '.join(test_result.get('groups_tested', []))
            }
            variance_results.append(result_dict)
    
    if variance_results:
        var_df = pd.DataFrame(variance_results)
        var_file = output_path / f"{prefix}_variance_tests.csv"
        var_df.to_csv(var_file, index=False)
        file_paths['variance'] = str(var_file)
        logger.info(f"Saved variance tests to {var_file}")
    
    # Recommendations
    rec_df = pd.DataFrame([assumption_results['recommendations']])
    rec_file = output_path / f"{prefix}_test_recommendations.csv"
    rec_df.to_csv(rec_file, index=False)
    file_paths['recommendations'] = str(rec_file)
    logger.info(f"Saved test recommendations to {rec_file}")
    
    return file_paths




