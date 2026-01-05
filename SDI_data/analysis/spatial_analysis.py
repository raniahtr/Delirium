"""
Spatial and regional analysis for SDI data.

This module performs regional aggregation, network-level analysis,
and spatial pattern analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import sys
import os

# Import from SDI_delirium utils
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from utils.atlas_labels import load_labels, load_correspondence, get_yeo_correspondance
except ImportError:
    # Fallback if not available
    load_labels = None
    load_correspondence = None
    get_yeo_correspondance = None

logger = logging.getLogger(__name__)


def get_parcel_region_category(parcel_id: int) -> str:
    """
    Get region category for a parcel ID.
    
    Categories:
    - Cortical: 1-360 (mapped to Yeo7 networks)
    - Subcortical: 361-424
    - Brainstem: 425-428
    - Cerebellum: 429-430
    
    Parameters:
    -----------
    parcel_id : int
        Parcel ID (0-indexed: 0-429)
    
    Returns:
    --------
    category : str
        Region category
    """
    # Convert to 1-indexed for category determination
    pid_1idx = parcel_id + 1
    
    if 1 <= pid_1idx <= 360:
        return 'Cortical'
    elif 361 <= pid_1idx <= 424:
        return 'Subcortical'
    elif 425 <= pid_1idx <= 428:
        return 'Brainstem'
    elif 429 <= pid_1idx <= 430:
        return 'Cerebellum'
    else:
        return 'Unknown'


def get_parcel_ordering_by_region(
    config: Optional[Dict] = None,
    parcel_to_yeo7: Optional[Dict[int, int]] = None,
    parcel_to_category: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    Get parcel ordering by brain region/network for spatial organization.
    
    Order: Cortical networks (Yeo7 1-7), Subcortical, Brainstem, Cerebellum
    Within each category, parcels are ordered by their ID.
    
    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary (required if mappings not provided)
    parcel_to_yeo7 : dict, optional
        Mapping from parcel_id (1-360) to Yeo7 network (1-7)
    parcel_to_category : dict, optional
        Mapping from parcel_id (361-430) to subcortical category
    
    Returns:
    --------
    ordering : np.ndarray
        Array of parcel IDs (0-indexed) in spatial order
    """
    # Try to load mappings if not provided
    if parcel_to_yeo7 is None or parcel_to_category is None:
        if config is not None and load_labels is not None:
            try:
                name_map, description_map = load_labels(config)
                
                # Build parcel_to_category from descriptions
                parcel_to_category = {}
                for pid in range(361, 431):  # Subcortical parcels
                    if pid in description_map:
                        desc = str(description_map[pid]).lower() if description_map[pid] else ''
                        if 'cerebellum' in desc or pid in (429, 430):
                            parcel_to_category[pid] = 'Cerebellum'
                        elif 'brainstem' in desc or pid in (425, 426, 427, 428):
                            parcel_to_category[pid] = 'Brainstem'
                        elif 'thalamus' in desc:
                            parcel_to_category[pid] = 'Thalamus'
                        elif 'amyg' in desc:
                            parcel_to_category[pid] = 'Amygdala'
                        elif any(k in desc for k in ['putamen', 'caudate', 'pallidus', 'accumbens']):
                            parcel_to_category[pid] = 'Basal Ganglia'
                        else:
                            parcel_to_category[pid] = 'Other'
                
                # Load Yeo7 correspondence for cortical parcels
                if load_correspondence is not None:
                    correspondence = load_correspondence(config)
                    parcel_to_yeo7 = {}
                    for pid in range(1, 361):  # Cortical parcels 1-360
                        if pid in correspondence:
                            networks = correspondence[pid]
                            if isinstance(networks, list) and len(networks) > 0:
                                parcel_to_yeo7[pid] = int(networks[0]) if networks[0] > 0 else None
                            elif isinstance(networks, (int, float)) and networks > 0:
                                parcel_to_yeo7[pid] = int(networks)
            except Exception as e:
                logger.warning(f"Could not load mappings for parcel ordering: {e}")
                parcel_to_yeo7 = {}
                parcel_to_category = {}
    
    # Build ordering
    ordered_parcels = []
    
    # 1. Cortical parcels by Yeo7 network (1-7)
    for network in range(1, 8):
        network_parcels = []
        for pid_1idx in range(1, 361):  # Cortical parcels
            if pid_1idx in parcel_to_yeo7 and parcel_to_yeo7[pid_1idx] == network:
                network_parcels.append(pid_1idx - 1)  # Convert to 0-indexed
        ordered_parcels.extend(sorted(network_parcels))
    
    # 2. Subcortical parcels (361-424)
    subcortical_parcels = list(range(360, 424))  # 0-indexed: 360-423
    ordered_parcels.extend(subcortical_parcels)
    
    # 3. Brainstem (425-428)
    brainstem_parcels = list(range(424, 428))  # 0-indexed: 424-427
    ordered_parcels.extend(brainstem_parcels)
    
    # 4. Cerebellum (429-430)
    cerebellum_parcels = list(range(428, 430))  # 0-indexed: 428-429
    ordered_parcels.extend(cerebellum_parcels)
    
    # Ensure all 430 parcels are included
    all_parcels = set(range(430))
    missing = all_parcels - set(ordered_parcels)
    if len(missing) > 0:
        logger.warning(f"Some parcels not included in ordering: {missing}")
        ordered_parcels.extend(sorted(missing))
    
    return np.array(ordered_parcels)


def aggregate_by_region(
    unified_df: pd.DataFrame,
    config: Optional[Dict] = None,
    parcel_to_yeo7: Optional[Dict[int, int]] = None,
    parcel_to_category: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Aggregate SDI values by brain region categories.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    config : dict, optional
        Configuration dictionary (required if parcel_to_yeo7/category not provided)
    parcel_to_yeo7 : dict, optional
        Mapping from parcel_id (1-360) to Yeo7 network (1-7)
    parcel_to_category : dict, optional
        Mapping from parcel_id (361-430) to subcortical category
    
    Returns:
    --------
    region_df : pd.DataFrame
        DataFrame with mean SDI per region per group
    """
    # Try to load mappings if not provided
    if parcel_to_yeo7 is None or parcel_to_category is None:
        if config is not None and load_labels is not None:
            try:
                # Load labels to get subcortical categories
                name_map, description_map = load_labels(config)
                
                # Build parcel_to_category from descriptions
                parcel_to_category = {}
                for pid, desc in description_map.items():
                    if pid >= 361 and pid <= 430:
                        desc_lower = str(desc).lower() if desc else ''
                        if 'cerebellum' in desc_lower or pid in (429, 430):
                            parcel_to_category[pid] = 'Cerebellum'
                        elif 'brainstem' in desc_lower or pid in (425, 426, 427, 428):
                            parcel_to_category[pid] = 'Brainstem'
                        elif 'thalamus' in desc_lower:
                            parcel_to_category[pid] = 'Thalamus'
                        elif 'amyg' in desc_lower:
                            parcel_to_category[pid] = 'Amygdala'
                        elif any(k in desc_lower for k in ['putamen', 'caudate', 'pallidus', 'accumbens']):
                            parcel_to_category[pid] = 'Basal Ganglia'
                        else:
                            parcel_to_category[pid] = 'Other'
                
                # Load Yeo7 correspondence for cortical parcels
                if load_correspondence is not None:
                    correspondence = load_correspondence(config)
                    parcel_to_yeo7 = {}
                    for pid in range(1, 361):  # Cortical parcels 1-360
                        if pid in correspondence:
                            networks = correspondence[pid]
                            # Take first network if list, or convert to int
                            if isinstance(networks, list) and len(networks) > 0:
                                parcel_to_yeo7[pid] = int(networks[0]) if networks[0] > 0 else None
                            elif isinstance(networks, (int, float)) and networks > 0:
                                parcel_to_yeo7[pid] = int(networks)
                
                logger.info("Loaded parcellation mappings from config")
            except Exception as e:
                logger.warning(f"Could not load parcellation mappings: {e}")
                parcel_to_yeo7 = {}
                parcel_to_category = {}
        else:
            parcel_to_yeo7 = {}
            parcel_to_category = {}
    
    # Add region information to dataframe
    unified_df = unified_df.copy()
    unified_df['region_category'] = unified_df['parcel_id'].apply(get_parcel_region_category)
    
    # For cortical parcels, add Yeo7 network
    unified_df['yeo7_network'] = None
    for idx, row in unified_df.iterrows():
        pid_1idx = int(row['parcel_id']) + 1  # Convert to 1-indexed
        if 1 <= pid_1idx <= 360 and pid_1idx in parcel_to_yeo7:
            unified_df.at[idx, 'yeo7_network'] = parcel_to_yeo7[pid_1idx]
        elif 361 <= pid_1idx <= 424 and pid_1idx in parcel_to_category:
            # For subcortical, use category
            unified_df.at[idx, 'yeo7_network'] = parcel_to_category[pid_1idx]
    
    # Aggregate by region and group
    region_stats = unified_df.groupby(['region_category', 'group'])['sdi_value'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    return region_stats


def aggregate_by_yeo7_network(
    unified_df: pd.DataFrame,
    config: Optional[Dict] = None,
    parcel_to_yeo7: Optional[Dict[int, int]] = None
) -> pd.DataFrame:
    """
    Aggregate SDI values by Yeo7 networks (cortical only).
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    config : dict, optional
        Configuration dictionary (required if parcel_to_yeo7 not provided)
    parcel_to_yeo7 : dict, optional
        Mapping from parcel_id (1-360) to Yeo7 network (1-7)
    
    Returns:
    --------
    network_df : pd.DataFrame
        DataFrame with mean SDI per Yeo7 network per group
    """
    # Try to load mappings if not provided
    if parcel_to_yeo7 is None:
        if config is not None and load_correspondence is not None:
            try:
                correspondence = load_correspondence(config)
                parcel_to_yeo7 = {}
                for pid in range(1, 361):  # Cortical parcels 1-360
                    if pid in correspondence:
                        networks = correspondence[pid]
                        # Take first network if list, or convert to int
                        if isinstance(networks, list) and len(networks) > 0:
                            parcel_to_yeo7[pid] = int(networks[0]) if networks[0] > 0 else None
                        elif isinstance(networks, (int, float)) and networks > 0:
                            parcel_to_yeo7[pid] = int(networks)
                logger.info("Loaded Yeo7 mappings from config")
            except Exception as e:
                logger.warning(f"Could not load Yeo7 mappings: {e}")
                parcel_to_yeo7 = {}
        else:
            parcel_to_yeo7 = {}
    
    # Filter to cortical parcels only (0-359 in 0-indexed, 1-360 in 1-indexed)
    cortical_df = unified_df[unified_df['parcel_id'] < 360].copy()
    
    # Add Yeo7 network
    cortical_df['yeo7_network'] = None
    for idx, row in cortical_df.iterrows():
        pid_1idx = int(row['parcel_id']) + 1  # Convert to 1-indexed
        if pid_1idx in parcel_to_yeo7 and parcel_to_yeo7[pid_1idx] is not None:
            cortical_df.at[idx, 'yeo7_network'] = parcel_to_yeo7[pid_1idx]
    
    # Filter out parcels without Yeo7 mapping
    cortical_df = cortical_df[cortical_df['yeo7_network'].notna()]
    
    if len(cortical_df) == 0:
        logger.warning("No cortical parcels with Yeo7 mapping found")
        return pd.DataFrame()
    
    # Aggregate by Yeo7 network and group
    network_stats = cortical_df.groupby(['yeo7_network', 'group'])['sdi_value'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    # Rename Yeo7 network IDs to names
    yeo7_names = {
        1: 'Visual',
        2: 'Somatomotor',
        3: 'DorsalAttn',
        4: 'VentralAttn',
        5: 'Limbic',
        6: 'FrontoParietal',
        7: 'Default'
    }
    network_stats['yeo7_network_name'] = network_stats['yeo7_network'].map(yeo7_names)
    
    return network_stats


def analyze_spatial_clustering(
    significance_map: np.ndarray,
    threshold: float = 0.05
) -> Dict:
    """
    Analyze spatial clustering of significant parcels.
    
    Parameters:
    -----------
    significance_map : np.ndarray
        Binary significance map (1 = significant, 0 = not)
    threshold : float
        Threshold for significance (default: 0.05)
    
    Returns:
    --------
    clustering_stats : dict
        Dictionary with clustering statistics
    """
    # Convert to binary if needed
    if significance_map.dtype != bool and significance_map.dtype != int:
        sig_binary = significance_map < threshold
    else:
        sig_binary = significance_map.astype(bool)
    
    n_significant = np.sum(sig_binary)
    n_total = len(sig_binary)
    
    # Simple clustering: count contiguous regions
    # This is a simplified version - for full spatial clustering,
    # would need adjacency information from the atlas
    clustering_stats = {
        'n_significant_parcels': int(n_significant),
        'n_total_parcels': int(n_total),
        'proportion_significant': float(n_significant / n_total) if n_total > 0 else 0.0,
        'spatial_clustering_note': 'Full spatial clustering requires atlas adjacency information'
    }
    
    return clustering_stats


def export_spatial_analysis(
    region_stats: pd.DataFrame,
    network_stats: pd.DataFrame,
    output_dir: str,
    prefix: str = "sdi"
) -> Dict[str, str]:
    """
    Export spatial analysis results to CSV.
    
    Parameters:
    -----------
    region_stats : pd.DataFrame
        Regional statistics
    network_stats : pd.DataFrame
        Network-level statistics
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
    
    # Regional statistics
    if len(region_stats) > 0:
        region_file = output_path / f"{prefix}_regional_stats.csv"
        region_stats.to_csv(region_file, index=False)
        file_paths['regional'] = str(region_file)
        logger.info(f"Exported regional statistics to {region_file}")
    
    # Network-level statistics
    if len(network_stats) > 0:
        network_file = output_path / f"{prefix}_yeo7_network_stats.csv"
        network_stats.to_csv(network_file, index=False)
        file_paths['yeo7_networks'] = str(network_file)
        logger.info(f"Exported Yeo7 network statistics to {network_file}")
    
    return file_paths


if __name__ == "__main__":
    # Test
    from .load_sdi_data import load_all_sdi_data
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.load_config import load_config
    
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    unified_df, _ = load_all_sdi_data(config)
    
    region_stats = aggregate_by_region(unified_df, config=config)
    print("\nRegional statistics:")
    print(region_stats)
    
    network_stats = aggregate_by_yeo7_network(unified_df, config=config)
    print("\nYeo7 network statistics:")
    print(network_stats)

