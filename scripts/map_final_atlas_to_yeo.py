#!/usr/bin/env python3
"""
Map Final_Combined_atlas_MNI2009c_1mm to Yeo 7 and Yeo 17 networks.

This script:
1. Loads Final_atlas (MNI2009c space) and parcels_labels.xlsx
2. Loads Yeo 7 and Yeo 17 networks
3. Resamples Yeo atlases to Final_atlas space (preserves parcel boundaries)
4. Maps each Final_atlas parcel to Yeo network with maximal voxel overlap
5. Handles subcortical extensions (labels 361-432)
6. Saves mapping tables to CSV

Usage:
    python3 map_final_atlas_to_yeo.py
"""

import nibabel as nib
import numpy as np
import pandas as pd
from collections import Counter
from nilearn.image import resample_to_img
import os
import sys
import re

# ============================================================================
# PATHS
# ============================================================================
ROOT = "/media/RCPNAS/Data/Delirium/Delirium_Rania"
ATLAS_DIR = os.path.join(ROOT, "atlas")

# Final atlas (MNI2009c space, 1mm)
FINAL_ATLAS_PATH = os.path.join(ATLAS_DIR, "Final_Combined_atlas_MNI2009c_1mm.nii.gz")
PARCELS_LABELS_XLSX = os.path.join(ATLAS_DIR, "parcels_label.xlsx")

# Yeo networks  
YEO7_PATH = os.path.join(ATLAS_DIR,"Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz")

# Output paths
OUTPUT_DIR = ATLAS_DIR
YEO7_RESAMP_PATH = os.path.join(OUTPUT_DIR, "yeo7_resamp_to_final_atlas.nii.gz")
LUT7_CSV = os.path.join(OUTPUT_DIR, "final_atlas_to_yeo7.csv")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_yeo_atlas(path_or_list):
    """Find the first available Yeo atlas path.
    
    Accepts either a single path (string) or a list of paths.
    """
    # Handle single string path
    if isinstance(path_or_list, str):
        if os.path.exists(path_or_list):
            return path_or_list
        return None
    
    # Handle list of paths
    for path in path_or_list:
        if os.path.exists(path):
            return path
    return None

def load_parcels_labels(xlsx_path):
    """Load parcels labels and Yeo mappings from Excel file.
    
    Returns:
    --------
    tuple: (id2name dict, id2yeo7 dict)
        id2name: parcel_id -> parcel_name
        id2yeo7: parcel_id -> yeo7_network (only for subcortical if column exists)
    """
    try:
        df = pd.read_excel(xlsx_path)
        # Try to identify columns (flexible)
        id_col = None
        name_col = None
        yeo_col = None
        
        # Common column name patterns
        for col in df.columns:
            col_lower = col.lower()
            if id_col is None and any(x in col_lower for x in ['id', 'index', 'label', 'parcel']):
                id_col = col
            if name_col is None and any(x in col_lower for x in ['name', 'region', 'label_name', 'parcel_name', 'area_name']):
                name_col = col
            if yeo_col is None and any(x in col_lower for x in ['yeo', 'network', 'yeo7', 'yeo_7']):
                yeo_col = col
        
        # Fallback: use first two columns
        if id_col is None:
            id_col = df.columns[0]
        if name_col is None and len(df.columns) > 1:
            name_col = df.columns[1]
        
        # Create mappings
        id2name = {}
        id2yeo7 = {}
        for _, row in df.iterrows():
            parcel_id = row[id_col]
            parcel_name = row[name_col] if name_col and pd.notna(row[name_col]) else f"Parcel_{parcel_id}"
            try:
                pid = int(parcel_id)
                id2name[pid] = str(parcel_name)
                
                # Load Yeo mapping if column exists and value is not NaN
                if yeo_col and pd.notna(row.get(yeo_col)):
                    try:
                        yeo_val = row[yeo_col]
                        # Handle both numeric and string values
                        if isinstance(yeo_val, (int, float)):
                            id2yeo7[pid] = int(yeo_val)
                        elif isinstance(yeo_val, str) and yeo_val.strip():
                            # Try to extract number from string
                            match = re.search(r'\d+', yeo_val)
                            if match:
                                id2yeo7[pid] = int(match.group())
                    except (ValueError, TypeError):
                        pass
            except (ValueError, TypeError):
                continue
        
        print(f"   Loaded {len(id2name)} parcel labels")
        print(f"  Example: Parcel 1 = {id2name.get(1, 'N/A')}")
        if id2yeo7:
            print(f"   Loaded {len(id2yeo7)} Yeo 7 mappings from Excel")
        return id2name, id2yeo7
    except Exception as e:
        print(f"  Warning: Could not load parcels labels: {e}")
        return {}, {}

def max_overlap_map(atlas_data, yeo_data, parcel_ids):
    """
    Map each parcel to the Yeo network with maximal voxel overlap.
    
    Parameters:
    -----------
    atlas_data : ndarray
        Final atlas data (integer labels)
    yeo_data : ndarray
        Yeo network data (integer labels 1-7 or 1-17)
    parcel_ids : list
        List of parcel IDs to map
    
    Returns:
    --------
    dict : Mapping from parcel_id -> yeo_network_id
    """
    lut = {}
    for parcel_id in parcel_ids:
        # Get voxels belonging to this parcel
        parcel_mask = (atlas_data == parcel_id)
        parcel_voxels = yeo_data[parcel_mask]
        
        # Filter out background (0)
        parcel_voxels = parcel_voxels[parcel_voxels > 0]
        
        if parcel_voxels.size > 0:
            # Find most common Yeo network
            most_common = Counter(parcel_voxels).most_common(1)[0]
            lut[parcel_id] = int(most_common[0])
        else:
            # No overlap with any Yeo network
            lut[parcel_id] = 0
    
    return lut

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Mapping Final_Combined_atlas to Yeo Networks")
    print("=" * 70)
    print()
    
    # Check inputs
    if not os.path.exists(FINAL_ATLAS_PATH):
        print(f"ERROR: Final atlas not found: {FINAL_ATLAS_PATH}")
        sys.exit(1)
    
    # Find Yeo atlases
    yeo7_path = find_yeo_atlas(YEO7_PATH)
    
    if yeo7_path is None:
        print(f"ERROR: Yeo 7 atlas not found. Checked:")
        print(f"  - {YEO7_PATH}")
        print("\nPlease download Yeo atlases or update paths in script.")
        sys.exit(1)
    
    
    print("Input files:")
    print(f"  Final atlas: {FINAL_ATLAS_PATH}")
    print(f"  Yeo 7: {yeo7_path}")
    print()
    
    # Load Final atlas
    print("Loading Final atlas...")
    final_atlas_img = nib.load(FINAL_ATLAS_PATH)
    final_atlas_data = final_atlas_img.get_fdata().astype(int)
    print(f"   Shape: {final_atlas_data.shape}")
    print(f"   Voxel size: {final_atlas_img.header.get_zooms()[:3]}")
    
    # Get unique parcel IDs (exclude background)
    parcel_ids = sorted([int(x) for x in np.unique(final_atlas_data) if x > 0])
    print(f"  ✓ Found {len(parcel_ids)} parcels (IDs: {parcel_ids[0]}-{parcel_ids[-1]})")
    print()
    
    # Load parcels labels and subcortical Yeo mappings
    print("Loading parcels labels...")
    id2name, id2yeo7_excel = load_parcels_labels(PARCELS_LABELS_XLSX)
    print()
    
    # Load Yeo atlases
    print("Loading Yeo atlases...")
    yeo7_img = nib.load(yeo7_path)
    print(f"  Yeo 7 shape: {yeo7_img.shape}, voxel size: {yeo7_img.header.get_zooms()[:3]}")
    print()
    
    # Resample Yeo to Final atlas space (preserves parcel boundaries)
    print("Resampling Yeo atlases to Final atlas space...")
    print("  (Using nearest-neighbor interpolation to preserve network labels)")
    
    if os.path.exists(YEO7_RESAMP_PATH):
        print(f"  Using existing: {YEO7_RESAMP_PATH}")
        yeo7_resamp = nib.load(YEO7_RESAMP_PATH)
    else:
        print("  Resampling Yeo 7...")
        yeo7_resamp = resample_to_img(yeo7_img, final_atlas_img, interpolation="nearest")
        nib.save(yeo7_resamp, YEO7_RESAMP_PATH)
        print(f"  ✓ Saved: {YEO7_RESAMP_PATH}")

    
    # Convert to arrays
    yeo7_data = yeo7_resamp.get_fdata().astype(int)
    
    # Separate cortical and subcortical parcels
    cortical_ids = [pid for pid in parcel_ids if 1 <= pid <= 360]
    subcortical_ids = [pid for pid in parcel_ids if 361 <= pid <= 432]
    
    print(f"  Cortical parcels (1-360): {len(cortical_ids)}")
    print(f"  Subcortical parcels (361-432): {len(subcortical_ids)}")
    print()
    
    # Create mappings
    print("Creating mappings...")
    print("  Mapping cortical parcels (1-360) to Yeo 7 using overlap method...")
    lut7_cortical = max_overlap_map(final_atlas_data, yeo7_data, cortical_ids)
    
    # For subcortical: use Excel mappings if available, otherwise 0
    print("  Mapping subcortical parcels (361-432) from Excel file...")
    lut7_subcortical = {}
    for pid in subcortical_ids:
        if pid in id2yeo7_excel:
            lut7_subcortical[pid] = id2yeo7_excel[pid]
        else:
            # No mapping in Excel, assign 0 (no Yeo network)
            lut7_subcortical[pid] = 0
    
    # Combine mappings
    lut7 = {**lut7_cortical, **lut7_subcortical}
   
    
    # Save mappings to CSV
    print("Saving mappings to CSV...")
    
    # Yeo 7 mapping
    rows7 = []
    for parcel_id in sorted(parcel_ids):
        rows7.append({
            'parcel_id': parcel_id,
            'parcel_name': id2name.get(parcel_id, f"Parcel_{parcel_id}"),
            'yeo7_network': lut7.get(parcel_id, 0),
            'yeo7_network_name': f"Network_{lut7.get(parcel_id, 0)}" if lut7.get(parcel_id, 0) > 0 else "None"
        })
    df7 = pd.DataFrame(rows7)
    df7.to_csv(LUT7_CSV, index=False)
    print(f"   Saved: {LUT7_CSV} ({len(df7)} rows)")
    
    
    # Summary statistics
    print("Summary:")
    print(f"  Total parcels: {len(parcel_ids)}")
    print(f"  Parcels mapped to Yeo 7: {sum(1 for v in lut7.values() if v > 0)}")
    
    # Check subcortical (labels 361-432)
    subcortical_ids = [pid for pid in parcel_ids if 361 <= pid <= 432]
    if subcortical_ids:
        print(f"\n  Subcortical parcels (361-432): {len(subcortical_ids)}")
        print("  Subcortical Yeo 7 mappings:")
        for pid in subcortical_ids:
            print(f"    Parcel {pid} ({id2name.get(pid, 'N/A')}): Yeo 7 network {lut7.get(pid, 0)}")
        
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()

