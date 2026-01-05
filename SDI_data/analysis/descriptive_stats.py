"""
Descriptive statistics for SDI data.

This module computes group-level and parcel-level descriptive statistics,
distribution analysis, and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def compute_group_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive group-level descriptive statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified SDI DataFrame with columns: ['parcel_id', 'sdi_value', 'group']
    
    Returns:
    --------
    stats_df : pd.DataFrame
        DataFrame with statistics per group
    """
    stats_list = []
    
    for group in df['group'].unique():
        group_data = df[df['group'] == group]['sdi_value'].values
        
        # Basic statistics
        stats_dict = {
            'group': group,
            'n_parcels': len(group_data),
            'mean': np.mean(group_data),
            'median': np.median(group_data),
            'std': np.std(group_data, ddof=1),
            'sem': stats.sem(group_data),
            'min': np.min(group_data),
            'max': np.max(group_data),
            'q25': np.percentile(group_data, 25),
            'q75': np.percentile(group_data, 75),
            'iqr': np.percentile(group_data, 75) - np.percentile(group_data, 25),
        }
        
        # Distribution statistics
        stats_dict['skewness'] = stats.skew(group_data)
        stats_dict['kurtosis'] = stats.kurtosis(group_data)
        
        # Confidence intervals (95%)
        ci = stats.t.interval(0.95, len(group_data)-1, 
                             loc=np.mean(group_data), 
                             scale=stats.sem(group_data))
        stats_dict['ci_lower'] = ci[0]
        stats_dict['ci_upper'] = ci[1]
        
        stats_list.append(stats_dict)
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df


def compute_parcel_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute parcel-level statistics (mean and median SDI per parcel per group).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    parcel_stats : pd.DataFrame
        DataFrame with parcel_id and statistics per group (mean, median, std, q25, q75)
    """
    # Group by parcel and group, compute statistics
    parcel_stats = df.groupby(['parcel_id', 'group'])['sdi_value'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('q25', lambda x: np.percentile(x, 25)),
        ('q75', lambda x: np.percentile(x, 75)),
        ('count', 'count')
    ]).reset_index()
    
    # Pivot to have groups as columns for mean
    parcel_pivot_mean = parcel_stats.pivot_table(
        values='mean',
        index='parcel_id',
        columns='group',
        aggfunc='first'
    ).reset_index()
    parcel_pivot_mean.columns = ['parcel_id'] + [f'mean_{col}' for col in parcel_pivot_mean.columns[1:]]
    
    # Pivot for median
    parcel_pivot_median = parcel_stats.pivot_table(
        values='median',
        index='parcel_id',
        columns='group',
        aggfunc='first'
    ).reset_index()
    parcel_pivot_median.columns = ['parcel_id'] + [f'median_{col}' for col in parcel_pivot_median.columns[1:]]
    
    # Pivot for std
    parcel_pivot_std = parcel_stats.pivot_table(
        values='std',
        index='parcel_id',
        columns='group',
        aggfunc='first'
    ).reset_index()
    parcel_pivot_std.columns = ['parcel_id'] + [f'std_{col}' for col in parcel_pivot_std.columns[1:]]
    
    # Pivot for q25
    parcel_pivot_q25 = parcel_stats.pivot_table(
        values='q25',
        index='parcel_id',
        columns='group',
        aggfunc='first'
    ).reset_index()
    parcel_pivot_q25.columns = ['parcel_id'] + [f'q25_{col}' for col in parcel_pivot_q25.columns[1:]]
    
    # Pivot for q75
    parcel_pivot_q75 = parcel_stats.pivot_table(
        values='q75',
        index='parcel_id',
        columns='group',
        aggfunc='first'
    ).reset_index()
    parcel_pivot_q75.columns = ['parcel_id'] + [f'q75_{col}' for col in parcel_pivot_q75.columns[1:]]
    
    # Merge all statistics
    parcel_pivot = parcel_pivot_mean.merge(parcel_pivot_median, on='parcel_id', how='outer')
    parcel_pivot = parcel_pivot.merge(parcel_pivot_std, on='parcel_id', how='outer')
    parcel_pivot = parcel_pivot.merge(parcel_pivot_q25, on='parcel_id', how='outer')
    parcel_pivot = parcel_pivot.merge(parcel_pivot_q75, on='parcel_id', how='outer')
    
    return parcel_pivot


def detect_outliers_iqr(df: pd.DataFrame, group: Optional[str] = None) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified SDI DataFrame
    group : str, optional
        If provided, detect outliers for specific group only
    
    Returns:
    --------
    outliers_df : pd.DataFrame
        DataFrame with outlier information
    """
    if group is not None:
        data_subset = df[df['group'] == group].copy()
    else:
        data_subset = df.copy()
    
    outliers_list = []
    
    for grp in data_subset['group'].unique():
        group_data = data_subset[data_subset['group'] == grp]
        
        Q1 = group_data['sdi_value'].quantile(0.25)
        Q3 = group_data['sdi_value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = group_data[
            (group_data['sdi_value'] < lower_bound) | 
            (group_data['sdi_value'] > upper_bound)
        ].copy()
        
        if len(outliers) > 0:
            outliers['outlier_type'] = np.where(
                outliers['sdi_value'] < lower_bound, 'lower', 'upper'
            )
            outliers['bound_value'] = np.where(
                outliers['sdi_value'] < lower_bound, lower_bound, upper_bound
            )
            outliers_list.append(outliers)
    
    if len(outliers_list) > 0:
        outliers_df = pd.concat(outliers_list, ignore_index=True)
    else:
        outliers_df = pd.DataFrame()
    
    return outliers_df


def compute_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution statistics for each group.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    dist_stats : pd.DataFrame
        Distribution statistics per group
    """
    dist_list = []
    
    for group in df['group'].unique():
        group_data = df[df['group'] == group]['sdi_value'].values
        
        # Normality tests
        if len(group_data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(group_data)
            dagostino_stat, dagostino_p = None, None
        else:
            shapiro_stat, shapiro_p = None, None
            dagostino_stat, dagostino_p = stats.normaltest(group_data)
        
        dist_dict = {
            'group': group,
            'n': len(group_data),
            'skewness': stats.skew(group_data),
            'kurtosis': stats.kurtosis(group_data),
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'dagostino_statistic': dagostino_stat,
            'dagostino_pvalue': dagostino_p,
        }
        
        dist_list.append(dist_dict)
    
    dist_stats = pd.DataFrame(dist_list)
    return dist_stats


def export_descriptive_stats(
    unified_df: pd.DataFrame,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export all descriptive statistics to CSV files.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_dir : str
        Output directory path
    prefix : str
        Prefix for output files
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping statistic type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Group-level statistics
    group_stats = compute_group_level_stats(unified_df)
    group_file = output_path / f"{prefix}_group_level_stats.csv"
    group_stats.to_csv(group_file, index=False)
    file_paths['group_level'] = str(group_file)
    logger.info(f"Exported group-level stats to {group_file}")
    
    # Parcel-level statistics
    parcel_stats = compute_parcel_level_stats(unified_df)
    parcel_file = output_path / f"{prefix}_parcel_level_stats.csv"
    parcel_stats.to_csv(parcel_file, index=False)
    file_paths['parcel_level'] = str(parcel_file)
    logger.info(f"Exported parcel-level stats to {parcel_file}")
    
    # Distribution statistics
    dist_stats = compute_distribution_stats(unified_df)
    dist_file = output_path / f"{prefix}_distribution_stats.csv"
    dist_stats.to_csv(dist_file, index=False)
    file_paths['distribution'] = str(dist_file)
    logger.info(f"Exported distribution stats to {dist_file}")
    
    # Outliers
    outliers = detect_outliers_iqr(unified_df)
    if len(outliers) > 0:
        outliers_file = output_path / f"{prefix}_outliers.csv"
        outliers.to_csv(outliers_file, index=False)
        file_paths['outliers'] = str(outliers_file)
        logger.info(f"Exported outliers to {outliers_file}")
    else:
        logger.info("No outliers detected")
        file_paths['outliers'] = None
    
    return file_paths


if __name__ == "__main__":
    # Test
    from load_sdi_data import load_all_sdi_data
    from utils.load_config import load_config
    
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    unified_df, _ = load_all_sdi_data(config)
    
    print("\nGroup-level statistics:")
    print(compute_group_level_stats(unified_df))
    
    print("\nDistribution statistics:")
    print(compute_distribution_stats(unified_df))

