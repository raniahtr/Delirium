"""
Statistical assumption verification for SDI data.

This module performs normality tests, homogeneity of variance tests,
and provides recommendations for appropriate statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, levene, bartlett

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
        'levene' (robust) or 'bartlett' (sensitive to non-normality)
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    # Prepare data arrays
    data_arrays = []
    group_names = []
    
    for group, data in group_data.items():
        data_clean = data[np.isfinite(data)]
        if len(data_clean) > 0:
            data_arrays.append(data_clean)
            group_names.append(group)
    
    if len(data_arrays) < 2:
        raise ValueError("Need at least 2 groups for variance test")
    
    # Perform test
    if test_type == 'levene':
        stat, pval = levene(*data_arrays)
        test_name = 'levene'
    elif test_type == 'bartlett':
        stat, pval = bartlett(*data_arrays)
        test_name = 'bartlett'
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    results = {
        'test_used': test_name,
        'statistic': stat,
        'pvalue': pval,
        'is_homogeneous': pval > 0.05,
        'groups_tested': group_names
    }
    
    return results


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
    variances_homogeneous = levene_result['is_homogeneous']
    
    recommendations = {
        'use_parametric': all_normal and variances_homogeneous,
        'use_nonparametric': not (all_normal and variances_homogeneous),
        'primary_test': 'kruskal_wallis' if not (all_normal and variances_homogeneous) else 'anova',
        'posthoc_test': 'mannwhitneyu' if not (all_normal and variances_homogeneous) else 'welch_ttest',
        'rationale': []
    }
    
    if not all_normal:
        recommendations['rationale'].append("Data is not normally distributed in at least one group")
    if not variances_homogeneous:
        recommendations['rationale'].append("Variances are not homogeneous across groups")
    if all_normal and variances_homogeneous:
        recommendations['rationale'].append("All assumptions met - parametric tests appropriate")
    
    results['recommendations'] = recommendations
    
    return results


def plot_qq_plots(
    unified_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create Q-Q plots for each group to assess normality.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    groups = sorted(unified_df['group'].unique())
    n_groups = len(groups)
    
    fig, axes = plt.subplots(1, n_groups, figsize=figsize)
    if n_groups == 1:
        axes = [axes]
    
    for idx, group in enumerate(groups):
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        
        stats.probplot(group_data_clean, dist="norm", plot=axes[idx])
        axes[idx].set_title(f'{group}\n(n={len(group_data_clean)})', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Q-Q plots to {output_path}")
    
    return fig


def plot_variance_comparison(
    unified_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    use_log_scale: bool = True,
    percentile_clip: Tuple[float, float] = (1, 99)
) -> plt.Figure:
    """
    Create boxplots to visually assess variance homogeneity.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    use_log_scale : bool
        Whether to use log scale for y-axis (default: True)
    percentile_clip : tuple
        Percentiles to use for y-axis limits (default: (1, 99))
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = sorted(unified_df['group'].unique())
    data_to_plot = [unified_df[unified_df['group'] == g]['sdi_value'].values 
                    for g in groups]
    
    # Calculate y-axis limits based on percentiles (excluding extreme outliers)
    all_data = np.concatenate(data_to_plot)
    all_data_clean = all_data[np.isfinite(all_data) & (all_data > 0)]
    
    if len(all_data_clean) > 0:
        if use_log_scale:
            # For log scale, use percentiles of log-transformed data
            log_data = np.log10(all_data_clean + 1e-10)
            y_min = np.percentile(log_data, percentile_clip[0])
            y_max = np.percentile(log_data, percentile_clip[1])
            y_label = 'SDI Value (log10)'
            # Transform data for plotting
            data_to_plot_transformed = [np.log10(d + 1e-10) for d in data_to_plot]
        else:
            # For linear scale, use percentiles of raw data
            y_min = np.percentile(all_data_clean, percentile_clip[0])
            y_max = np.percentile(all_data_clean, percentile_clip[1])
            y_label = 'SDI Value'
            # Add some padding
            y_range = y_max - y_min
            y_min = max(0, y_min - 0.05 * y_range)
            y_max = y_max + 0.05 * y_range
            data_to_plot_transformed = data_to_plot
    else:
        y_min, y_max = 0, 1
        y_label = 'SDI Value'
        data_to_plot_transformed = data_to_plot
    
    bp = ax.boxplot(data_to_plot_transformed, labels=groups, patch_artist=True, 
                    showfliers=False)  # Hide outliers for cleaner view
    
    # Color boxes (using group colors)
    from utils.group import get_group_color
    colors = get_group_color()
    color_map = {
        'ICU': colors.get('CR', '#E67E22'),
        'ICU Delirium': colors.get('Delirium', '#C0392B'),
        'Healthy Controls': colors.get('HC', '#3498DB')
    }
    
    for patch, group in zip(bp['boxes'], groups):
        patch.set_facecolor(color_map.get(group, '#95A5A6'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_title('Variance Comparison Across Groups', fontsize=14, fontweight='bold')
    ax.set_ylim(y_min, y_max)
    if use_log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved variance comparison plot to {output_path}")
    
    return fig


def export_assumption_checks(
    assumption_results: Dict,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export assumption check results to CSV.
    
    Parameters:
    -----------
    assumption_results : dict
        Results from check_all_assumptions
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
    
    # Variance tests
    variance_list = []
    for test_name, test_result in assumption_results['variance_tests'].items():
        variance_list.append({
            'test': test_name,
            'statistic': test_result['statistic'],
            'pvalue': test_result['pvalue'],
            'is_homogeneous': test_result['is_homogeneous'],
            'groups': ', '.join(test_result['groups_tested'])
        })
    
    if variance_list:
        var_df = pd.DataFrame(variance_list)
        var_file = output_path / f"{prefix}_variance_tests.csv"
        var_df.to_csv(var_file, index=False)
        file_paths['variance'] = str(var_file)
    
    # Recommendations
    rec_df = pd.DataFrame([assumption_results['recommendations']])
    rec_file = output_path / f"{prefix}_test_recommendations.csv"
    rec_df.to_csv(rec_file, index=False)
    file_paths['recommendations'] = str(rec_file)
    
    return file_paths


if __name__ == "__main__":
    # Test
    from load_sdi_data import load_all_sdi_data
    from utils.load_config import load_config
    
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    unified_df, _ = load_all_sdi_data(config)
    
    results = check_all_assumptions(unified_df)
    print("\nAssumption check results:")
    print(f"Recommendations: {results['recommendations']}")

