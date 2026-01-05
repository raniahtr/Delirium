"""
Load SDI data from h5 files for all groups.

This module loads SDI h5 files, creates unified DataFrames, and handles
group name mapping and data validation.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Group name mapping: internal names -> standard names
GROUP_NAME_MAPPING = {
    'CR': 'ICU',
    'Delirium': 'ICU Delirium',
    'HC': 'Healthy Controls',
    'all': 'All'
}

# Reverse mapping for file loading
REVERSE_GROUP_MAPPING = {v: k for k, v in GROUP_NAME_MAPPING.items()}


def load_individual_sdi_data_from_ref(
    file_path: str,
    config: Dict,
    internal_group: str
) -> pd.DataFrame:
    """
    Load individual subject SDI data from SDI_ref format file.
    
    The SDI_ref format has:
    - Rows: 430 parcels (index 0-429)
    - Columns: Subject IDs (e.g., 'sub-01', 'sub-02', ...)
    - Values: SDI for each parcel-subject combination
    
    Parameters:
    -----------
    file_path : str
        Path to the SDI_ref_{group}_{band}.h5 file
    config : dict
        Configuration dictionary (needed for subject-to-group mapping)
    internal_group : str
        Internal group name (CR, Delirium, HC) - used as fallback if mapping fails
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with columns: ['subject_id', 'parcel_id', 'sdi_value', 'group']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SDI_ref file not found: {file_path}")
    
    try:
        # Load DataFrame with subjects as columns
        df_wide = pd.read_hdf(file_path, key='SDI')
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")
    
    # Verify structure: should have subjects as columns
    if df_wide.shape[0] != 430:
        raise ValueError(f"Expected 430 parcels, got {df_wide.shape[0]}")
    
    # Get subject IDs from columns
    subject_ids = df_wide.columns.tolist()
    
    # Import group mapping function
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.group import get_group
    except ImportError:
        logger.warning("Could not import utils.group.get_group, using fallback group assignment")
        get_group = None
    
    # Melt from wide to long format
    df_long = df_wide.reset_index().melt(
        id_vars=['index'],
        value_vars=subject_ids,
        var_name='subject_id',
        value_name='sdi_value'
    )
    
    # Rename index column to parcel_id
    df_long.rename(columns={'index': 'parcel_id'}, inplace=True)
    
    # Map subjects to groups
    groups = []
    unmapped_subjects = []
    
    for subject_id in df_long['subject_id'].unique():
        try:
            if get_group is not None:
                # get_group expects 'sub-XX' format and converts internally
                internal_group_name = get_group(config, subject_id)
                # Map internal group name to display name
                display_group_name = GROUP_NAME_MAPPING.get(internal_group_name, internal_group_name)
            else:
                # Fallback: use the group from filename
                display_group_name = GROUP_NAME_MAPPING.get(internal_group, internal_group)
            
            groups.append(display_group_name)
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not map subject {subject_id} to group: {e}, using fallback")
            display_group_name = GROUP_NAME_MAPPING.get(internal_group, internal_group)
            groups.append(display_group_name)
            unmapped_subjects.append(subject_id)
    
    # Create mapping dictionary
    subject_to_group = dict(zip(df_long['subject_id'].unique(), groups))
    
    # Add group column
    df_long['group'] = df_long['subject_id'].map(subject_to_group)
    
    if unmapped_subjects:
        logger.warning(f"Could not map {len(unmapped_subjects)} subjects to groups, using fallback")
    
    # Select and reorder columns
    df_long = df_long[['subject_id', 'parcel_id', 'sdi_value', 'group']].copy()
    
    # Log summary
    n_subjects = len(subject_ids)
    n_parcels = 430
    logger.info(f"Loaded {n_subjects} subjects × {n_parcels} parcels = {len(df_long)} rows from {file_path}")
    
    # Log sample subject-to-group mapping
    sample_subjects = df_long['subject_id'].unique()[:5]
    sample_mapping = {subj: subject_to_group[subj] for subj in sample_subjects}
    logger.info(f"Sample subject-to-group mapping: {sample_mapping}")
    
    return df_long


def load_sdi_h5_file(file_path: str, group_name: str) -> pd.DataFrame:
    """
    Load SDI data from a single h5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5 file
    group_name : str
        Standard group name (ICU, ICU Delirium, Healthy Controls)
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with columns: ['parcel_id', 'sdi_value', 'group']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SDI file not found: {file_path}")
    
    try:
        df = pd.read_hdf(file_path, key='SDI')
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")
    
    # Ensure we have the right structure
    if 'SDI' not in df.columns:
        raise ValueError(f"Expected 'SDI' column in {file_path}")
    
    # Create parcel_id column (0-indexed)
    df['parcel_id'] = df.index.values
    df['sdi_value'] = df['SDI'].values
    df['group'] = group_name
    
    # Select only needed columns
    df = df[['parcel_id', 'sdi_value', 'group']].copy()
    
    return df


def _load_from_new_pipeline(
    config: Dict,
    band: str,
    pipeline_type: str = 'HC_ICU',
    ref_group: str = 'HC'
) -> Optional[pd.DataFrame]:
    """
    Load SDI data from new pipeline file structure.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with paths
    band : str
        Frequency band
    pipeline_type : str
        Pipeline type: 'HC_ICU', 'ref_group', or 'per_group' (default: 'HC_ICU')
    ref_group : str
        Reference group name (required when pipeline_type='ref_group', default: 'HC')
    
    Returns:
    --------
    df : pd.DataFrame or None
        Unified DataFrame with columns: ['subject_id', 'parcel_id', 'sdi_value', 'group']
        Returns None if file not found
    """
    results_dir = config['dir']['results_dir']
    
    # Construct file path based on pipeline_type
    if pipeline_type == 'HC_ICU':
        file_path = os.path.join(results_dir, 'SDI_HC_ICU', band, f'SDI_HC_ICU_{band}.h5')
    elif pipeline_type == 'ref_group':
        file_path = os.path.join(results_dir, 'SDI_ref_group', ref_group, band, f'SDI_ref_group_{band}.h5')
    elif pipeline_type == 'per_group':
        file_path = os.path.join(results_dir, 'SDI_per_group', band, f'SDI_per_group_{band}.h5')
    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}. Must be 'HC_ICU', 'ref_group', or 'per_group'")
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.debug(f"New pipeline file not found: {file_path}")
        return None
    
    try:
        # Load DataFrame with subjects as columns (same structure as old format)
        df_wide = pd.read_hdf(file_path, key='SDI')
        
        # Verify structure: should have subjects as columns
        if df_wide.shape[0] != 430:
            raise ValueError(f"Expected 430 parcels, got {df_wide.shape[0]}")
        
        # Get subject IDs from columns
        subject_ids = df_wide.columns.tolist()
        
        # Import group mapping function
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.group import get_group
        except ImportError:
            logger.warning("Could not import utils.group.get_group, using fallback group assignment")
            get_group = None
        
        # Melt from wide to long format
        df_long = df_wide.reset_index().melt(
            id_vars=['index'],
            value_vars=subject_ids,
            var_name='subject_id',
            value_name='sdi_value'
        )
        
        # Rename index column to parcel_id
        df_long.rename(columns={'index': 'parcel_id'}, inplace=True)
        
        # Map subjects to groups
        groups = []
        unmapped_subjects = []
        
        for subject_id in df_long['subject_id'].unique():
            try:
                if get_group is not None:
                    # get_group expects 'sub-XX' format and converts internally
                    internal_group_name = get_group(config, subject_id)
                    # Map internal group name to display name
                    display_group_name = GROUP_NAME_MAPPING.get(internal_group_name, internal_group_name)
                else:
                    # Fallback: cannot determine group without get_group function
                    logger.warning(f"Cannot map subject {subject_id} to group without get_group function")
                    display_group_name = 'Unknown'
                
                groups.append(display_group_name)
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not map subject {subject_id} to group: {e}")
                display_group_name = 'Unknown'
                groups.append(display_group_name)
                unmapped_subjects.append(subject_id)
        
        # Create mapping dictionary
        subject_to_group = dict(zip(df_long['subject_id'].unique(), groups))
        
        # Add group column
        df_long['group'] = df_long['subject_id'].map(subject_to_group)
        
        if unmapped_subjects:
            logger.warning(f"Could not map {len(unmapped_subjects)} subjects to groups")
        
        # Select and reorder columns
        df_long = df_long[['subject_id', 'parcel_id', 'sdi_value', 'group']].copy()
        
        # Log summary
        n_subjects = len(subject_ids)
        n_parcels = 430
        logger.info(f"Loaded {n_subjects} subjects × {n_parcels} parcels = {len(df_long)} rows from new pipeline: {file_path}")
        
        # Log sample subject-to-group mapping
        sample_subjects = df_long['subject_id'].unique()[:5]
        sample_mapping = {subj: subject_to_group[subj] for subj in sample_subjects}
        logger.info(f"Sample subject-to-group mapping: {sample_mapping}")
        
        return df_long
        
    except Exception as e:
        logger.warning(f"Error loading from new pipeline file {file_path}: {e}")
        return None


def load_all_sdi_data(
    config: Dict, 
    band: str = 'bandpass',
    use_individual_data: bool = True,
    pipeline_type: str = 'HC_ICU',
    ref_group: str = 'HC'
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load SDI data for all groups and create unified structures.
    
    First tries to load from new pipeline structure (HC_ICU, ref_group, or per_group).
    Falls back to old structure (SDI_ref_* or SDI_* files) if new pipeline files don't exist.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with paths
    band : str
        Frequency band (default: 'bandpass')
    use_individual_data : bool
        If True, try to load individual subject data from SDI_ref files first.
        If False, only load group-averaged data. (default: True)
    pipeline_type : str
        Pipeline type to load from: 'HC_ICU', 'ref_group', or 'per_group' (default: 'HC_ICU')
    ref_group : str
        Reference group name (required when pipeline_type='ref_group', default: 'HC')
    
    Returns:
    --------
    unified_df : pd.DataFrame
        Unified DataFrame with columns: ['subject_id', 'parcel_id', 'sdi_value', 'group']
        (subject_id may be missing if using group-averaged data)
    group_arrays : dict
        Dictionary mapping group names to numpy arrays.
        For individual data: (n_subjects, 430) arrays
        For group-averaged data: (430,) arrays
    """
    results_dir = config['dir']['results_dir']
    
    # Standard group names
    standard_groups = ['ICU', 'ICU Delirium', 'Healthy Controls']
    
    # First attempt: Try loading from new pipeline
    if use_individual_data:
        df_new = _load_from_new_pipeline(config, band, pipeline_type, ref_group)
        if df_new is not None:
            using_individual_data = True
            logger.info(f"Successfully loaded data from new pipeline (type: {pipeline_type})")
            
            # Create group arrays: (n_subjects, 430) for each group
            group_arrays = {}
            for std_group in standard_groups:
                group_df = df_new[df_new['group'] == std_group]
                if len(group_df) > 0:
                    subjects = group_df['subject_id'].unique()
                    n_subjects = len(subjects)
                    group_array = np.zeros((n_subjects, 430))
                    
                    for i, subject in enumerate(subjects):
                        subject_data = group_df[group_df['subject_id'] == subject].sort_values('parcel_id')
                        if len(subject_data) == 430:
                            group_array[i, :] = subject_data['sdi_value'].values
                        else:
                            logger.warning(f"Subject {subject} in group {std_group} has {len(subject_data)} parcels, expected 430")
                    
                    group_arrays[std_group] = group_array
            
            # Validate data
            validate_sdi_data(df_new, group_arrays, has_subject_id=using_individual_data)
            
            # Log subject counts per group
            for group in df_new['group'].unique():
                n_subjects = df_new[df_new['group'] == group]['subject_id'].nunique()
                n_parcels = df_new[df_new['group'] == group]['parcel_id'].nunique()
                logger.info(f"  {group}: {n_subjects} subjects × {n_parcels} parcels")
            
            return df_new, group_arrays
        else:
            logger.info("New pipeline file not found, falling back to old format")
    
    # Fallback: Load from old structure
    sdi_dir = os.path.join(results_dir, 'SDI')
    
    # Load data for each group
    all_dfs = []
    group_arrays = {}
    using_individual_data = False
    
    for std_group in standard_groups:
        # Map to internal group name for file lookup
        internal_group = REVERSE_GROUP_MAPPING.get(std_group)
        if internal_group is None:
            logger.warning(f"No mapping found for group: {std_group}, skipping")
            continue
        
        df = None
        
        # Try to load individual subject data first (if requested)
        if use_individual_data:
            ref_file_name = f'SDI_ref_{internal_group}_{band}.h5'
            ref_file_path = os.path.join(sdi_dir, ref_file_name)
            
            if os.path.exists(ref_file_path):
                try:
                    df = load_individual_sdi_data_from_ref(ref_file_path, config, internal_group)
                    using_individual_data = True
                    logger.info(f"Loaded individual SDI data for {std_group} from {ref_file_name}")
                    
                    # Create group array: (n_subjects, 430)
                    # Pivot to get subjects × parcels matrix
                    subjects = df['subject_id'].unique()
                    n_subjects = len(subjects)
                    group_array = np.zeros((n_subjects, 430))
                    
                    for i, subject in enumerate(subjects):
                        subject_data = df[df['subject_id'] == subject].sort_values('parcel_id')
                        group_array[i, :] = subject_data['sdi_value'].values
                    
                    group_arrays[std_group] = group_array
                    all_dfs.append(df)
                    continue
                except Exception as e:
                    logger.warning(f"Error loading individual data from {ref_file_path}: {e}, trying group-averaged")
        
        # Fall back to group-averaged data
        file_name = f'SDI_{internal_group}_{band}.h5'
        file_path = os.path.join(sdi_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.warning(f"SDI file not found for {std_group}: {file_path}")
            continue
        
        try:
            df = load_sdi_h5_file(file_path, std_group)
            all_dfs.append(df)
            
            # Store as array (430 values)
            group_arrays[std_group] = df['sdi_value'].values
            
            logger.info(f"Loaded group-averaged SDI data for {std_group}: {len(df)} parcels")
        except Exception as e:
            logger.error(f"Error loading {std_group}: {e}")
            continue
    
    if len(all_dfs) == 0:
        raise ValueError("No SDI data files could be loaded")
    
    # Combine all DataFrames
    unified_df = pd.concat(all_dfs, ignore_index=True)
    
    # Validate data
    validate_sdi_data(unified_df, group_arrays, has_subject_id=using_individual_data)
    
    data_type = "individual subject" if using_individual_data else "group-averaged"
    logger.info(f"Successfully loaded {data_type} SDI data for {len(standard_groups)} groups, "
                f"{len(unified_df)} total rows")
    
    if using_individual_data:
        # Log subject counts per group
        for group in unified_df['group'].unique():
            n_subjects = unified_df[unified_df['group'] == group]['subject_id'].nunique()
            n_parcels = unified_df[unified_df['group'] == group]['parcel_id'].nunique()
            logger.info(f"  {group}: {n_subjects} subjects × {n_parcels} parcels")
    
    return unified_df, group_arrays


def validate_sdi_data(
    df: pd.DataFrame, 
    group_arrays: Dict[str, np.ndarray],
    has_subject_id: bool = False
) -> None:
    """
    Validate SDI data integrity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified SDI DataFrame
    group_arrays : dict
        Dictionary of group arrays
    has_subject_id : bool
        Whether the DataFrame contains individual subject data (has 'subject_id' column)
    """
    # Check for missing values
    n_missing = df['sdi_value'].isna().sum()
    if n_missing > 0:
        logger.warning(f"Found {n_missing} missing SDI values")
    
    # Check for infinite values
    n_inf = np.isinf(df['sdi_value']).sum()
    if n_inf > 0:
        logger.warning(f"Found {n_inf} infinite SDI values")
    
    # Check for negative values (SDI should be positive)
    n_negative = (df['sdi_value'] < 0).sum()
    if n_negative > 0:
        logger.warning(f"Found {n_negative} negative SDI values")
    
    # Check expected number of parcels (430)
    expected_parcels = 430
    n_parcels = df['parcel_id'].nunique()
    if n_parcels != expected_parcels:
        logger.warning(f"Found {n_parcels} unique parcels, expected {expected_parcels}")
    
    # Validate group arrays
    for group, arr in group_arrays.items():
        if has_subject_id:
            # Individual data: should be (n_subjects, 430)
            if arr.ndim != 2:
                logger.warning(f"Group {group} array should be 2D (n_subjects, 430), got shape {arr.shape}")
            elif arr.shape[1] != expected_parcels:
                logger.warning(f"Group {group} has {arr.shape[1]} parcels, expected {expected_parcels}")
            else:
                n_subjects_in_array = arr.shape[0]
                n_subjects_in_df = df[df['group'] == group]['subject_id'].nunique() if has_subject_id else 1
                if n_subjects_in_array != n_subjects_in_df:
                    logger.warning(f"Group {group}: array has {n_subjects_in_array} subjects, "
                                 f"DataFrame has {n_subjects_in_df} subjects")
        else:
            # Group-averaged data: should be (430,)
            if len(arr) != expected_parcels:
                logger.warning(f"Group {group} has {len(arr)} parcels, expected {expected_parcels}")
    
    # Check subject-to-group mapping (if individual data)
    if has_subject_id and 'subject_id' in df.columns:
        # Check that all subjects in a group are correctly assigned
        for group in df['group'].unique():
            group_subjects = df[df['group'] == group]['subject_id'].unique()
            logger.debug(f"Group {group}: {len(group_subjects)} subjects")
            
            # Check for duplicate subject-parcel combinations
            duplicates = df[df['group'] == group].duplicated(subset=['subject_id', 'parcel_id']).sum()
            if duplicates > 0:
                logger.warning(f"Group {group}: Found {duplicates} duplicate subject-parcel combinations")
    
    # Check for outliers using IQR method
    for group in df['group'].unique():
        group_data = df[df['group'] == group]['sdi_value']
        Q1 = group_data.quantile(0.25)
        Q3 = group_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((group_data < lower_bound) | (group_data > upper_bound)).sum()
        if outliers > 0:
            logger.info(f"Group {group}: {outliers} potential outliers detected (IQR method)")


def get_group_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract group-level summary data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    group_summary : pd.DataFrame
        Summary statistics per group
    """
    group_summary = df.groupby('group')['sdi_value'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
        ('count', 'count')
    ]).reset_index()
    
    # Calculate IQR
    group_summary['iqr'] = group_summary['q75'] - group_summary['q25']
    
    return group_summary


def get_parcel_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract parcel-level data with group means.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified SDI DataFrame
    
    Returns:
    --------
    parcel_df : pd.DataFrame
        DataFrame with parcel_id and mean SDI per group
    """
    # Pivot to get mean SDI per parcel per group
    parcel_df = df.pivot_table(
        values='sdi_value',
        index='parcel_id',
        columns='group',
        aggfunc='mean'
    ).reset_index()
    
    return parcel_df


if __name__ == "__main__":
    # Test loading
    from utils.load_config import load_config
    
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    unified_df, group_arrays = load_all_sdi_data(config)
    print("\nUnified DataFrame shape:", unified_df.shape)
    print("\nGroups loaded:", list(group_arrays.keys()))
    print("\nGroup-level summary:")
    print(get_group_level_data(unified_df))


