"""
Spatial analysis for SDI data.

This module performs regional and network-level aggregation and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging
from scipy.stats import kruskal, f_oneway

logger = logging.getLogger(__name__)


def aggregate_by_region(
    unified_df: pd.DataFrame,
    atlas_to_yeo7_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate SDI values by region category.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    atlas_to_yeo7_df : pd.DataFrame
        DataFrame with parcel_id and yeo7_network columns
    
    Returns:
    --------
    region_stats : pd.DataFrame
        DataFrame with regional statistics per group
    """
    from utils.connectome_utils import get_region_category_indices
    
    logger.info("Building parcel-to-region mapping...")
    
    # Pre-compute parcel-to-region mapping (cache to avoid repeated calls)
    unique_parcels = unified_df['parcel_id'].unique()
    parcel_to_region = {}
    for parcel_id in unique_parcels:
        region_cat = get_region_category_indices(parcel_id, atlas_to_yeo7_df)
        if region_cat is not None:
            parcel_to_region[parcel_id] = region_cat
    
    # Add region_category column to unified_df for efficient grouping
    unified_df = unified_df.copy()
    unified_df['region_category'] = unified_df['parcel_id'].map(parcel_to_region)
    
    # Filter out rows where region_category is None
    unified_df = unified_df.dropna(subset=['region_category'])
    
    if len(unified_df) == 0:
        logger.warning("No data with valid region categories")
        return pd.DataFrame()
    
    region_categories = sorted(unified_df['region_category'].unique())
    logger.info(f"Aggregating by {len(region_categories)} region categories...")
    
    region_stats_list = []
    
    # Use pandas groupby for efficient aggregation
    for (group, region_cat), group_region_df in unified_df.groupby(['group', 'region_category']):
        # Get SDI values for this region
        region_data = group_region_df['sdi_value'].values
        region_data_clean = region_data[np.isfinite(region_data)]
        
        if len(region_data_clean) == 0:
            continue
        
        # Get unique parcels in this region
        n_parcels = group_region_df['parcel_id'].nunique()
        
        # Compute statistics
        stats_dict = {
            'group': group,
            'region_category': region_cat,
            'n_parcels': n_parcels,
            'n_observations': len(region_data_clean),
            'mean': np.mean(region_data_clean),
            'median': np.median(region_data_clean),
            'std': np.std(region_data_clean, ddof=1),
            'sem': np.std(region_data_clean, ddof=1) / np.sqrt(len(region_data_clean)),
            'min': np.min(region_data_clean),
            'max': np.max(region_data_clean),
            'q25': np.percentile(region_data_clean, 25),
            'q75': np.percentile(region_data_clean, 75)
        }
        
        region_stats_list.append(stats_dict)
    
    region_stats = pd.DataFrame(region_stats_list)
    logger.info(f"Computed statistics for {len(region_stats)} group-region combinations")
    return region_stats


def perform_regional_comparisons(
    region_stats_df: pd.DataFrame,
    use_parametric: bool = False
) -> pd.DataFrame:
    """
    Perform statistical comparisons at regional level.
    
    Parameters:
    -----------
    region_stats_df : pd.DataFrame
        DataFrame with regional statistics (from aggregate_by_region)
    use_parametric : bool
        Whether to use parametric tests (default: False)
    
    Returns:
    --------
    regional_comparisons : pd.DataFrame
        DataFrame with regional comparison results
    """
    # Need to go back to original data for proper statistical tests
    # This function would need access to unified_df, so we'll handle it differently
    # For now, return empty - this can be enhanced later
    return pd.DataFrame()


def aggregate_by_yeo7_network(
    unified_df: pd.DataFrame,
    atlas_to_yeo7_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate SDI values by Yeo7 network.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    atlas_to_yeo7_df : pd.DataFrame
        DataFrame with parcel_id and yeo7_network columns
    
    Returns:
    --------
    network_stats : pd.DataFrame
        DataFrame with network-level statistics per group
    """
    from utils.connectome_utils import get_region_category_indices
    
    logger.info("Building parcel-to-region mapping for Yeo7 networks...")
    
    # Yeo7 network names
    yeo7_names = {
        1: 'Visual',
        2: 'Somatomotor',
        3: 'Dorsal Attention',
        4: 'Ventral Attention',
        5: 'Limbic',
        6: 'Frontoparietal',
        7: 'Default Mode'
    }
    
    # Pre-compute parcel-to-region mapping (cache to avoid repeated calls)
    unique_parcels = unified_df['parcel_id'].unique()
    parcel_to_region = {}
    for parcel_id in unique_parcels:
        region_cat = get_region_category_indices(parcel_id, atlas_to_yeo7_df)
        if region_cat is not None:
            parcel_to_region[parcel_id] = region_cat
    
    # Filter to only cortical Yeo7 networks (exclude subcortical, brainstem, cerebellum)
    yeo7_network_names = list(yeo7_names.values())
    
    # Add region_category column to unified_df
    unified_df = unified_df.copy()
    unified_df['region_category'] = unified_df['parcel_id'].map(parcel_to_region)
    
    # Filter to only Yeo7 networks
    unified_df = unified_df[unified_df['region_category'].isin(yeo7_network_names)]
    
    if len(unified_df) == 0:
        logger.warning("No data with valid Yeo7 network categories")
        return pd.DataFrame()
    
    logger.info(f"Aggregating by {len(yeo7_network_names)} Yeo7 networks...")
    
    network_stats_list = []
    
    # Use pandas groupby for efficient aggregation
    for (group, region_cat), group_network_df in unified_df.groupby(['group', 'region_category']):
        # Get SDI values for this network
        network_data = group_network_df['sdi_value'].values
        network_data_clean = network_data[np.isfinite(network_data)]
        
        if len(network_data_clean) == 0:
            continue
        
        # Get network number from name
        network_num = None
        for num, name in yeo7_names.items():
            if name == region_cat:
                network_num = num
                break
        
        if network_num is None:
            continue
        
        # Get unique parcels in this network
        n_parcels = group_network_df['parcel_id'].nunique()
        
        # Compute statistics
        stats_dict = {
            'group': group,
            'yeo7_network': network_num,
            'region_category': region_cat,
            'n_parcels': n_parcels,
            'n_observations': len(network_data_clean),
            'mean': np.mean(network_data_clean),
            'median': np.median(network_data_clean),
            'std': np.std(network_data_clean, ddof=1),
            'sem': np.std(network_data_clean, ddof=1) / np.sqrt(len(network_data_clean)),
            'min': np.min(network_data_clean),
            'max': np.max(network_data_clean)
        }
        
        network_stats_list.append(stats_dict)
    
    network_stats = pd.DataFrame(network_stats_list)
    logger.info(f"Computed statistics for {len(network_stats)} group-network combinations")
    return network_stats


def perform_regional_statistical_tests(
    unified_df: pd.DataFrame,
    atlas_to_yeo7_df: pd.DataFrame,
    use_parametric: bool = False
) -> pd.DataFrame:
    """
    Perform statistical tests at regional level.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    atlas_to_yeo7_df : pd.DataFrame
        DataFrame with parcel_id and yeo7_network columns
    use_parametric : bool
        Whether to use parametric tests (default: False)
    
    Returns:
    --------
    regional_tests : pd.DataFrame
        DataFrame with regional test results
    """
    from utils.connectome_utils import get_region_category_indices
    
    # Get all unique region categories
    region_categories = set()
    
    for parcel_id in unified_df['parcel_id'].unique():
        region_cat = get_region_category_indices(parcel_id, atlas_to_yeo7_df)
        if region_cat is not None:
            region_categories.add(region_cat)
    
    region_categories = sorted(region_categories)
    groups = sorted(unified_df['group'].unique())
    
    test_results = []
    
    logger.info(f"Performing regional statistical tests for {len(region_categories)} regions...")
    
    for region_cat in region_categories:
        # Get parcels in this region
        region_parcels = []
        for parcel_id in unified_df['parcel_id'].unique():
            parcel_region = get_region_category_indices(parcel_id, atlas_to_yeo7_df)
            if parcel_region == region_cat:
                region_parcels.append(parcel_id)
        
        if len(region_parcels) == 0:
            continue
        
        # Get data for each group
        group_data_list = []
        valid_groups = []
        
        for group in groups:
            group_df = unified_df[unified_df['group'] == group]
            region_data = group_df[group_df['parcel_id'].isin(region_parcels)]['sdi_value'].values
            region_data_clean = region_data[np.isfinite(region_data)]
            
            if len(region_data_clean) > 0:
                group_data_list.append(region_data_clean)
                valid_groups.append(group)
        
        if len(group_data_list) < 2:
            continue
        
        # Perform test
        if use_parametric:
            stat, pval = f_oneway(*group_data_list)
            test_name = 'ANOVA'
        else:
            stat, pval = kruskal(*group_data_list)
            test_name = 'Kruskal-Wallis'
        
        test_results.append({
            'region_category': region_cat,
            'test': test_name,
            'statistic': stat,
            'pvalue': pval,
            'n_groups': len(valid_groups),
            'groups': ', '.join(valid_groups)
        })
    
    regional_tests = pd.DataFrame(test_results)
    return regional_tests


def export_regional_stats(
    region_stats_df: pd.DataFrame,
    regional_tests_df: Optional[pd.DataFrame] = None,
    output_dir: str = None,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export regional statistics to CSV files.
    
    Parameters:
    -----------
    region_stats_df : pd.DataFrame
        DataFrame with regional statistics
    regional_tests_df : pd.DataFrame, optional
        DataFrame with regional test results
    output_dir : str
        Output directory
    prefix : str
        File prefix
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping result type to file path
    """
    if output_dir is None:
        return {}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Regional statistics
    if len(region_stats_df) > 0:
        stats_file = output_path / f"{prefix}_regional_stats.csv"
        region_stats_df.to_csv(stats_file, index=False)
        file_paths['regional_stats'] = str(stats_file)
        logger.info(f"Saved regional statistics to {stats_file}")
    
    # Regional tests
    if regional_tests_df is not None and len(regional_tests_df) > 0:
        tests_file = output_path / f"{prefix}_regional_tests.csv"
        regional_tests_df.to_csv(tests_file, index=False)
        file_paths['regional_tests'] = str(tests_file)
        logger.info(f"Saved regional tests to {tests_file}")
    
    return file_paths

