"""
Load SDI data from SDI_HC_ICU h5 files for analysis.

This module loads SDI h5 files from the SDI_HC_ICU pipeline, creates unified DataFrames,
and handles group name mapping.
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


def load_sdi_hc_icu_data(
    config: Dict,
    band: str = 'bandpass',
    pipeline_type: str = 'HC_ICU'
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load SDI data from SDI_HC_ICU h5 file structure.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with paths
    band : str
        Frequency band (default: 'bandpass')
    pipeline_type : str
        Pipeline type: 'HC_ICU' (default)
    
    Returns:
    --------
    unified_df : pd.DataFrame
        Unified DataFrame with columns: ['subject_id', 'parcel_id', 'sdi_value', 'group']
    group_arrays : dict
        Dictionary mapping group names to numpy arrays (n_subjects, 430)
    """
    results_dir = Path(config.get('results_dir', 'results'))
    
    # Construct file path
    if pipeline_type == 'HC_ICU':
        file_path = results_dir / 'SDI_HC_ICU' / band / f'SDI_HC_ICU_{band}.h5'
    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}. Must be 'HC_ICU'")
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"SDI file not found: {file_path}")
    
    logger.info(f"Loading SDI data from: {file_path}")
    
    try:
        # Load DataFrame with subjects as columns (430 rows × n_subjects columns)
        df_wide = pd.read_hdf(file_path, key='SDI')
        
        # Verify structure: should have subjects as columns
        if df_wide.shape[0] != 430:
            raise ValueError(f"Expected 430 parcels, got {df_wide.shape[0]}")
        
        # Get subject IDs from columns (format: 'sub-01', 'sub-02', etc.)
        subject_ids = df_wide.columns.tolist()
        logger.info(f"Found {len(subject_ids)} subjects")
        logger.debug(f"Sample subject IDs: {subject_ids[:5]}")
        
        # Load group assignments from config
        group_assignments = _load_group_assignments(config, subject_ids)
        
        # Create unified DataFrame
        all_rows = []
        group_arrays = {}
        
        # Standard group names
        standard_groups = ['ICU', 'ICU Delirium', 'Healthy Controls']
        
        for std_group in standard_groups:
            # Map to internal group name
            internal_group = REVERSE_GROUP_MAPPING.get(std_group)
            if internal_group is None:
                continue
            
            # Get subjects for this group
            group_subjects = group_assignments.get(internal_group, [])
            if len(group_subjects) == 0:
                logger.warning(f"No subjects found for group: {std_group}")
                continue
            
            # Filter subjects that are in the SDI data
            # Handle both hyphen and underscore formats
            # SDI file uses hyphens (sub-01), Excel uses underscores (sub_01)
            group_subjects_in_data = []
            for s in group_subjects:
                # Try exact match first
                if s in subject_ids:
                    group_subjects_in_data.append(s)
                else:
                    # Try converting underscore to hyphen (Excel -> SDI format)
                    s_hyphen = s.replace('_', '-')
                    if s_hyphen in subject_ids:
                        group_subjects_in_data.append(s_hyphen)  # Use hyphen format for DataFrame access
                    else:
                        # Try converting hyphen to underscore (SDI -> Excel format)
                        s_underscore = s.replace('-', '_')
                        for sid in subject_ids:
                            if sid.replace('-', '_') == s_underscore:
                                group_subjects_in_data.append(sid)  # Use original SDI format
                                break
            
            logger.info(f"{std_group}: {len(group_subjects_in_data)} subjects in SDI data")
            
            if len(group_subjects_in_data) == 0:
                continue
            
            # Create group array: (n_subjects, 430)
            n_subjects = len(group_subjects_in_data)
            group_array = np.zeros((n_subjects, 430))
            
            # Extract data for each subject
            for i, subject_id in enumerate(group_subjects_in_data):
                subject_data = df_wide[subject_id].values
                
                # Create rows for this subject
                for parcel_id in range(1, 431):  # Parcels are 1-indexed
                    all_rows.append({
                        'subject_id': subject_id,
                        'parcel_id': parcel_id,
                        'sdi_value': subject_data[parcel_id - 1],  # Convert to 0-indexed
                        'group': std_group
                    })
                
                # Store in group array
                group_array[i, :] = subject_data
            
            group_arrays[std_group] = group_array
        
        # Create unified DataFrame
        unified_df = pd.DataFrame(all_rows)
        
        if len(unified_df) == 0:
            raise ValueError(
                "No data loaded! Check that:\n"
                "1. Group assignments are correctly loaded from config\n"
                "2. Subject IDs in SDI file match those in group assignments\n"
                "3. Config has 'group' key pointing to Excel file with 'sub_ID' and 'Group' columns"
            )
        
        logger.info(f"Loaded SDI data: {len(unified_df)} rows, {len(group_arrays)} groups")
        for group, arr in group_arrays.items():
            logger.info(f"  {group}: {arr.shape[0]} subjects × {arr.shape[1]} parcels")
        
        return unified_df, group_arrays
        
    except Exception as e:
        logger.error(f"Error loading SDI data: {e}")
        raise


def _load_group_assignments(config: Dict, subject_ids_from_file: list = None) -> Dict[str, list]:
    """
    Load group assignments from config.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    group_assignments : dict
        Dictionary mapping internal group names to lists of subject IDs
    """
    try:
        # Try to use existing utils.group functions if config has 'group' key
        # This matches the pattern used in utils.group._recover_data
        if 'group' in config:
            try:
                from utils.group import load_group_assignments
                # load_group_assignments returns dict with group names as keys and arrays as values
                group_results = load_group_assignments('subjects_underscore', config)
                
                # Convert to our format
                # group_results keys are: 'CR', 'Delirium', 'HC' (from Excel)
                # values are numpy arrays of subject IDs in underscore format
                group_assignments = {}
                for group_name, subject_array in group_results.items():
                    # Group names from Excel are already internal names (CR, Delirium, HC)
                    if group_name in GROUP_NAME_MAPPING:
                        # Convert numpy array to list
                        group_assignments[group_name] = subject_array.tolist()
                
                if group_assignments:
                    total_subjects = sum(len(v) for v in group_assignments.values())
                    logger.info(f"Loaded group assignments from utils.group: {total_subjects} subjects across {len(group_assignments)} groups")
                    for group, subjects in group_assignments.items():
                        logger.debug(f"  {group}: {len(subjects)} subjects (sample: {subjects[:3]})")
                    return group_assignments
            except Exception as e:
                logger.warning(f"Could not use utils.group functions: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Try to load from group annotations Excel file
        group_file = config.get('group_annotations')
        if group_file and os.path.exists(group_file):
            group_df = pd.read_excel(group_file)
            
            # Try to find subject ID and group columns
            sub_col = None
            group_col = None
            
            for col in group_df.columns:
                col_lower = col.lower()
                if sub_col is None and any(x in col_lower for x in ['sub', 'subject', 'id']):
                    sub_col = col
                if group_col is None and any(x in col_lower for x in ['group']):
                    group_col = col
            
            if sub_col and group_col:
                group_assignments = {}
                for _, row in group_df.iterrows():
                    # Keep subject ID in underscore format for matching
                    subject_id_raw = str(row[sub_col])
                    subject_id = subject_id_raw.replace('-', '_')
                    group = str(row[group_col]).strip()
                    
                    # Map to internal group name if needed
                    if group not in GROUP_NAME_MAPPING:
                        # Try reverse mapping
                        if group in REVERSE_GROUP_MAPPING.values():
                            # Already standard name, find internal
                            for int_name, std_name in GROUP_NAME_MAPPING.items():
                                if std_name == group:
                                    group = int_name
                                    break
                    
                    if group in GROUP_NAME_MAPPING:
                        if group not in group_assignments:
                            group_assignments[group] = []
                        group_assignments[group].append(subject_id)
                
                if group_assignments:
                    logger.info(f"Loaded group assignments from Excel: {sum(len(v) for v in group_assignments.values())} subjects across {len(group_assignments)} groups")
                    return group_assignments
    except Exception as e:
        logger.warning(f"Could not load group assignments from file: {e}")
    
    # Fallback: try to infer from subject IDs
    # This is a simple heuristic - subjects starting with certain patterns
    # You may need to adjust this based on your naming convention
    logger.warning("Using fallback group assignment - may need manual configuration")
    return {}


def load_atlas_to_yeo7_mapping(
    atlas_to_yeo7_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load atlas to Yeo7 network mapping.
    
    Parameters:
    -----------
    atlas_to_yeo7_csv : Path, optional
        Path to CSV file (default: from connectome_utils)
    
    Returns:
    --------
    atlas_to_yeo7_df : pd.DataFrame
        DataFrame with columns 'parcel_id' and 'yeo7_network'
    """
    from utils.connectome_utils import ATLAS_TO_YEO7_CSV
    
    if atlas_to_yeo7_csv is None:
        atlas_to_yeo7_csv = ATLAS_TO_YEO7_CSV
    
    if not atlas_to_yeo7_csv.exists():
        raise FileNotFoundError(f"Atlas to Yeo7 mapping file not found: {atlas_to_yeo7_csv}")
    
    df = pd.read_csv(atlas_to_yeo7_csv)
    logger.info(f"Loaded atlas to Yeo7 mapping: {len(df)} parcels")
    
    return df
