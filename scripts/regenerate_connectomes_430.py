#!/usr/bin/env python3
"""
Regenerate connectome CSV files by removing parcels 390 and 423.

This script processes all existing connectome CSV files (432×432) and creates
new 430×430 versions by removing rows/columns 389 and 422 (0-indexed for 
parcels 390 and 423). The original files are backed up with a .backup extension.

Usage:
    python regenerate_connectomes_430.py [--preproc-dir PREPROC_DIR] [--dry-run]
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import shutil
from typing import List

# Parcels to exclude (1-indexed)
EXCLUDED_PARCELS = [390, 423]
# Corresponding matrix indices (0-indexed)
EXCLUDED_INDICES = [389, 422]

# Common connectome types to process
CONNECTOME_TYPES = [
    'SC_sift2',
    'SC_sift2_sizecorr',
    'COUNT',
    'TOTAL_length',
    'MEAN_length',
    'MEAN_FA',
    'MEAN_MD',
    'MEAN_AD',
    'MEAN_RD',
    'SC_invlen_sum'
]


def remove_excluded_parcels(matrix: np.ndarray) -> np.ndarray:
    """
    Remove rows and columns corresponding to excluded parcels.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input 432×432 matrix
        
    Returns:
    --------
    np.ndarray
        430×430 matrix with excluded rows/columns removed
    """
    if matrix.shape != (432, 432):
        raise ValueError(f"Expected 432×432 matrix, got {matrix.shape}")
    
    # Create mask to keep all indices except excluded ones
    keep_indices = [i for i in range(432) if i not in EXCLUDED_INDICES]
    
    # Remove rows and columns
    reduced_matrix = matrix[np.ix_(keep_indices, keep_indices)]
    
    if reduced_matrix.shape != (430, 430):
        raise ValueError(f"Expected 430×430 output, got {reduced_matrix.shape}")
    
    return reduced_matrix


def process_connectome_file(csv_path: Path, dry_run: bool = False) -> bool:
    """
    Process a single connectome CSV file.
    
    Parameters:
    -----------
    csv_path : Path
        Path to the connectome CSV file
    dry_run : bool
        If True, only report what would be done without making changes
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Load the matrix
        matrix = np.loadtxt(csv_path, delimiter=",")
        
        # Check if it's already 430×430
        if matrix.shape == (430, 430):
            print(f"  ⚠️  Already 430×430: {csv_path.name}")
            return True
        
        # Check if it's 432×432
        if matrix.shape != (432, 432):
            print(f"  ⚠️  Unexpected shape {matrix.shape}: {csv_path.name}")
            return False
        
        # Remove excluded parcels
        reduced_matrix = remove_excluded_parcels(matrix)
        
        if dry_run:
            print(f"  [DRY RUN] Would process: {csv_path.name} (432×432 → 430×430)")
            return True
        
        # Create backup
        backup_path = csv_path.with_suffix(csv_path.suffix + '.backup')
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)
            print(f"  ✓ Backed up: {backup_path.name}")
        
        # Save the reduced matrix
        np.savetxt(csv_path, reduced_matrix, delimiter=",", fmt="%.6g")
        print(f"  ✓ Processed: {csv_path.name} (432×432 → 430×430)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {csv_path.name}: {e}")
        return False


def find_connectome_files(preproc_dir: Path) -> List[Path]:
    """
    Find all connectome CSV files in the preprocessing directory.
    
    Parameters:
    -----------
    preproc_dir : Path
        Path to preprocessing directory
        
    Returns:
    --------
    List[Path]
        List of connectome CSV file paths
    """
    connectome_files = []
    
    # Find all subject directories
    for subj_dir in sorted(preproc_dir.glob("sub-*")):
        if not subj_dir.is_dir():
            continue
        
        connectome_local = subj_dir / "connectome_local"
        if not connectome_local.exists():
            continue
        
        # Find connectome files for this subject
        for connectome_type in CONNECTOME_TYPES:
            # Pattern: sub-XX_connectome_type.csv
            pattern = f"{subj_dir.name}_{connectome_type}.csv"
            connectome_file = connectome_local / pattern
            
            if connectome_file.exists():
                connectome_files.append(connectome_file)
    
    return connectome_files


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate connectome CSV files by removing parcels 390 and 423"
    )
    parser.add_argument(
        "--preproc-dir",
        type=str,
        default="/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_current",
        help="Path to preprocessing directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    preproc_dir = Path(args.preproc_dir)
    if not preproc_dir.exists():
        print(f"Error: Preprocessing directory not found: {preproc_dir}")
        return 1
    
    print("=" * 70)
    print("REGENERATE CONNECTOMES: Remove Parcels 390 and 423")
    print("=" * 70)
    print(f"Preprocessing directory: {preproc_dir}")
    print(f"Excluded parcels: {EXCLUDED_PARCELS}")
    print(f"Excluded indices (0-indexed): {EXCLUDED_INDICES}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Find all connectome files
    print("Finding connectome files...")
    connectome_files = find_connectome_files(preproc_dir)
    print(f"Found {len(connectome_files)} connectome files")
    print()
    
    if len(connectome_files) == 0:
        print("No connectome files found!")
        return 1
    
    # Process each file
    print("Processing files...")
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for csv_path in connectome_files:
        result = process_connectome_file(csv_path, dry_run=args.dry_run)
        if result:
            if "Already 430×430" in str(csv_path):
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already 430×430): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {len(connectome_files)}")
    
    if args.dry_run:
        print()
        print("This was a dry run. Run without --dry-run to apply changes.")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())

