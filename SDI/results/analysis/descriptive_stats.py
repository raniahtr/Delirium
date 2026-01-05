"""
Descriptive statistics for SDI data.

This module computes group-level, parcel-level, and distribution statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def compute_group_level_stats(unified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute group-level descriptive statistics.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame with columns: ['subject_id', 'parcel_id', 'sdi_value', 'group']
    
    Returns:
    --------
    group_stats : pd.DataFrame
        DataFrame with group-level statistics
    """
    group_stats_list = []
    
    for group in unified_df['group'].unique():
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        
        n_subjects = unified_df[unified_df['group'] == group]['subject_id'].nunique()
        n_parcels = unified_df[unified_df['group'] == group]['parcel_id'].nunique()
        
        if len(group_data_clean) == 0:
            continue
        
        stats_dict = {
            'group': group,
            'n_subjects': n_subjects,
            'n_parcels': n_parcels,
            'n_observations': len(group_data_clean),
            'mean': np.mean(group_data_clean),
            'median': np.median(group_data_clean),
            'std': np.std(group_data_clean, ddof=1),
            'sem': stats.sem(group_data_clean),
            'min': np.min(group_data_clean),
            'max': np.max(group_data_clean),
            'q25': np.percentile(group_data_clean, 25),
            'q75': np.percentile(group_data_clean, 75),
            'iqr': np.percentile(group_data_clean, 75) - np.percentile(group_data_clean, 25),
            'skewness': stats.skew(group_data_clean),
            'kurtosis': stats.kurtosis(group_data_clean)
        }
        
        group_stats_list.append(stats_dict)
    
    group_stats = pd.DataFrame(group_stats_list)
    return group_stats


def compute_parcel_level_stats(unified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute parcel-level descriptive statistics per group.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    parcel_stats : pd.DataFrame
        DataFrame with parcel-level statistics per group
    """
    parcel_stats_list = []
    
    for group in unified_df['group'].unique():
        group_df = unified_df[unified_df['group'] == group]
        
        for parcel_id in sorted(group_df['parcel_id'].unique()):
            parcel_data = group_df[group_df['parcel_id'] == parcel_id]['sdi_value'].values
            parcel_data_clean = parcel_data[np.isfinite(parcel_data)]
            
            if len(parcel_data_clean) == 0:
                continue
            
            stats_dict = {
                'group': group,
                'parcel_id': parcel_id,
                'n_subjects': len(parcel_data_clean),
                'mean': np.mean(parcel_data_clean),
                'median': np.median(parcel_data_clean),
                'std': np.std(parcel_data_clean, ddof=1),
                'sem': stats.sem(parcel_data_clean),
                'min': np.min(parcel_data_clean),
                'max': np.max(parcel_data_clean),
                'q25': np.percentile(parcel_data_clean, 25),
                'q75': np.percentile(parcel_data_clean, 75),
                'iqr': np.percentile(parcel_data_clean, 75) - np.percentile(parcel_data_clean, 25)
            }
            
            parcel_stats_list.append(stats_dict)
    
    parcel_stats = pd.DataFrame(parcel_stats_list)
    return parcel_stats


def detect_outliers_iqr(unified_df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    factor : float
        IQR factor for outlier detection (default: 1.5)
    
    Returns:
    --------
    outliers_df : pd.DataFrame
        DataFrame with detected outliers
    """
    outliers_list = []
    
    for group in unified_df['group'].unique():
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        
        if len(group_data_clean) == 0:
            continue
        
        q1 = np.percentile(group_data_clean, 25)
        q3 = np.percentile(group_data_clean, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Find outliers in the full group data
        group_df = unified_df[unified_df['group'] == group]
        outlier_mask = (group_df['sdi_value'] < lower_bound) | (group_df['sdi_value'] > upper_bound)
        outliers = group_df[outlier_mask].copy()
        
        if len(outliers) > 0:
            outliers['outlier_type'] = 'IQR'
            outliers['lower_bound'] = lower_bound
            outliers['upper_bound'] = upper_bound
            outliers_list.append(outliers)
    
    if len(outliers_list) > 0:
        outliers_df = pd.concat(outliers_list, ignore_index=True)
    else:
        outliers_df = pd.DataFrame()
    
    return outliers_df


def compute_distribution_stats(unified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution statistics per group.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    dist_stats : pd.DataFrame
        DataFrame with distribution statistics
    """
    dist_stats_list = []
    
    for group in unified_df['group'].unique():
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        
        if len(group_data_clean) == 0:
            continue
        
        # Outlier detection
        q1 = np.percentile(group_data_clean, 25)
        q3 = np.percentile(group_data_clean, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = np.sum((group_data_clean < lower_bound) | (group_data_clean > upper_bound))
        
        stats_dict = {
            'group': group,
            'n_observations': len(group_data_clean),
            'skewness': stats.skew(group_data_clean),
            'kurtosis': stats.kurtosis(group_data_clean),
            'n_outliers_iqr': n_outliers,
            'outlier_percentage': (n_outliers / len(group_data_clean)) * 100 if len(group_data_clean) > 0 else 0
        }
        
        dist_stats_list.append(stats_dict)
    
    dist_stats = pd.DataFrame(dist_stats_list)
    return dist_stats


def export_descriptive_stats(
    unified_df: pd.DataFrame,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Compute and export all descriptive statistics to CSV files.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_dir : str
        Output directory for CSV files
    prefix : str
        File prefix (default: "sdi")
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping statistic type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Group-level stats
    logger.info("Computing group-level statistics...")
    group_stats = compute_group_level_stats(unified_df)
    group_file = output_path / f"{prefix}_group_level_stats.csv"
    group_stats.to_csv(group_file, index=False)
    file_paths['group_level'] = str(group_file)
    logger.info(f"Saved group-level stats to {group_file}")
    
    # Parcel-level stats
    logger.info("Computing parcel-level statistics...")
    parcel_stats = compute_parcel_level_stats(unified_df)
    parcel_file = output_path / f"{prefix}_parcel_level_stats.csv"
    parcel_stats.to_csv(parcel_file, index=False)
    file_paths['parcel_level'] = str(parcel_file)
    logger.info(f"Saved parcel-level stats to {parcel_file}")
    
    # Distribution stats
    logger.info("Computing distribution statistics...")
    dist_stats = compute_distribution_stats(unified_df)
    dist_file = output_path / f"{prefix}_distribution_stats.csv"
    dist_stats.to_csv(dist_file, index=False)
    file_paths['distribution'] = str(dist_file)
    logger.info(f"Saved distribution stats to {dist_file}")
    
    # Outliers
    logger.info("Detecting outliers...")
    outliers = detect_outliers_iqr(unified_df)
    if len(outliers) > 0:
        outliers_file = output_path / f"{prefix}_outliers.csv"
        outliers.to_csv(outliers_file, index=False)
        file_paths['outliers'] = str(outliers_file)
        logger.info(f"Saved outliers to {outliers_file}")
    else:
        logger.info("No outliers detected")
        file_paths['outliers'] = None
    
    return file_paths




