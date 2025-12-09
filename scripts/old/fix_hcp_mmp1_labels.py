#!/usr/bin/env python3
"""
Fix HCP-MMP1 atlas labels: Remap right hemisphere labels from 1-180 to 181-360
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import argparse


def get_hemisphere_mask(data, affine, hemisphere='right'):
    """
    Create a mask for left or right hemisphere based on MNI coordinates.
    
    In MNI space, the right hemisphere typically has positive X coordinates
    (or negative, depending on orientation). We check the affine matrix to
    determine the orientation.
    
    Parameters:
    -----------
    data : numpy array
        The 3D label image
    affine : numpy array
        The affine transformation matrix (4x4)
    hemisphere : str
        'right' or 'left'
    
    Returns:
    --------
    mask : numpy array (boolean)
        True for voxels in the specified hemisphere
    """
    # Get voxel coordinates
    shape = data.shape
    coords = np.meshgrid(np.arange(shape[0]), 
                        np.arange(shape[1]), 
                        np.arange(shape[2]), 
                        indexing='ij')
    voxel_coords = np.stack([coords[0].ravel(), 
                            coords[1].ravel(), 
                            coords[2].ravel(), 
                            np.ones(np.prod(shape))])
    
    # Convert to world coordinates (MNI space)
    world_coords = affine @ voxel_coords
    x_coords = world_coords[0, :].reshape(shape)
    
    # In MNI space, right hemisphere typically has positive X
    # Check the sign of the X-axis in the affine matrix
    x_axis_sign = np.sign(affine[0, 0])
    
    if hemisphere == 'right':
        # Right hemisphere: positive X in MNI space
        if x_axis_sign > 0:
            mask = x_coords > 0
        else:
            mask = x_coords < 0
    else:  # left
        # Left hemisphere: negative X in MNI space
        if x_axis_sign > 0:
            mask = x_coords < 0
        else:
            mask = x_coords > 0
    
    return mask


def fix_hcp_mmp1_labels(nifti_path, output_path=None, label_file_path=None, dry_run=False):
    """
    Fix HCP-MMP1 atlas by remapping right hemisphere labels from 1-180 to 181-360.
    
    Parameters:
    -----------
    nifti_path : str or Path
        Path to the input NIfTI file
    output_path : str or Path, optional
        Path to save the corrected NIfTI file. If None, overwrites input.
    label_file_path : str or Path, optional
        Path to the label text file for verification (informational only)
    dry_run : bool
        If True, only show what would be changed without saving
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"Input NIfTI file not found: {nifti_path}")
    
    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 70)
        print()
    
    # Load the NIfTI file
    print(f"Loading NIfTI file: {nifti_path}")
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    affine = img.affine
    
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    
    # Round floating point labels to integers (common issue after transformations)
    # Check if labels are close to integers
    data_nonzero = data[data > 0]
    if len(data_nonzero) > 0:
        # Check if values are close to integers (within 0.1)
        is_close_to_int = np.allclose(data_nonzero, np.round(data_nonzero), atol=0.1)
        if not is_close_to_int:
            print(f"    Warning: Labels appear to be floating point, not integers")
            print(f"    Rounding to nearest integers...")
        # Round to integers
        data = np.round(data).astype(np.int32)
        data[data < 0] = 0  # Ensure no negative values
    
    # Get unique labels
    unique_labels = np.unique(data[data > 0])
    print(f"  Unique labels: {len(unique_labels)}")
    print(f"  Label range: {int(unique_labels.min())} to {int(unique_labels.max())}")
    
    # Check if labels are already correct (should have 1-360)
    if unique_labels.max() <= 180:
        print("\n⚠ WARNING: Maximum label is <= 180. This suggests overlapping labels.")
        print("  The atlas likely has both hemispheres using labels 1-180.")
        print("  We need to remap the right hemisphere to 181-360.\n")
    elif unique_labels.max() == 360 and unique_labels.min() == 1:
        print("\n✓ Labels appear to be in the correct range (1-360).")
        print("  However, we'll still check and fix if needed.\n")
    
    # Create right hemisphere mask
    print("Identifying right hemisphere voxels...")
    right_hemisphere_mask = get_hemisphere_mask(data, affine, hemisphere='right')
    left_hemisphere_mask = get_hemisphere_mask(data, affine, hemisphere='left')
    
    # Count voxels in each hemisphere
    right_voxels = np.sum(right_hemisphere_mask & (data > 0))
    left_voxels = np.sum(left_hemisphere_mask & (data > 0))
    print(f"  Left hemisphere labeled voxels: {left_voxels:,}")
    print(f"  Right hemisphere labeled voxels: {right_voxels:,}")
    
    # Analyze labels in each hemisphere
    left_labels = np.unique(data[left_hemisphere_mask & (data > 0)])
    right_labels = np.unique(data[right_hemisphere_mask & (data > 0)])
    
    print(f"\n  Label analysis by hemisphere:")
    print(f"    Left hemisphere: {len(left_labels)} unique labels")
    if len(left_labels) > 0:
        print(f"      Range: {int(left_labels.min())} to {int(left_labels.max())}")
        if len(left_labels) <= 20:
            print(f"      Labels: {sorted(left_labels.astype(int).tolist())}")
    
    print(f"    Right hemisphere: {len(right_labels)} unique labels")
    if len(right_labels) > 0:
        print(f"      Range: {int(right_labels.min())} to {int(right_labels.max())}")
        if len(right_labels) <= 20:
            print(f"      Labels: {sorted(right_labels.astype(int).tolist())}")
    
    # Check for overlapping labels (same label number in both hemispheres)
    overlapping_labels = np.intersect1d(left_labels, right_labels)
    if len(overlapping_labels) > 0:
        print(f"\n   OVERLAPPING LABELS DETECTED:")
        print(f"    {len(overlapping_labels)} labels appear in BOTH hemispheres")
        if len(overlapping_labels) <= 20:
            print(f"    Overlapping labels: {sorted(overlapping_labels.astype(int).tolist())}")
        else:
            print(f"    Examples: {sorted(overlapping_labels.astype(int).tolist())[:10]}...")
        print(f"    This confirms the need for remapping!\n")
    else:
        print(f"\n   No overlapping labels found (each label appears in only one hemisphere)")
    
    # Create a copy of the data for remapping
    data_fixed = data.copy()
    
    # Find right hemisphere voxels with labels 1-180
    right_labels_1_180 = right_hemisphere_mask & (data > 0) & (data <= 180)
    num_to_remap = np.sum(right_labels_1_180)
    
    if num_to_remap > 0:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Remapping {num_to_remap:,} right hemisphere voxels from 1-180 to 181-360...")
        
        # Get the labels that need to be remapped
        labels_to_remap = data[right_labels_1_180]
        unique_labels_to_remap = np.unique(labels_to_remap)
        
        print(f"  Labels being remapped: {len(unique_labels_to_remap)} unique label(s)")
        if len(unique_labels_to_remap) <= 20:
            print(f"    {sorted(unique_labels_to_remap.astype(int).tolist())}")
        else:
            print(f"    Range: {int(unique_labels_to_remap.min())} to {int(unique_labels_to_remap.max())}")
            print(f"    Examples: {sorted(unique_labels_to_remap.astype(int).tolist())[:10]}...")
        
        # Show remapping examples
        print(f"\n  Remapping examples (first 10):")
        for i, old_label in enumerate(sorted(unique_labels_to_remap.astype(int).tolist())[:10]):
            new_label = old_label + 180
            count = np.sum(labels_to_remap == old_label)
            print(f"    Label {old_label} -> {new_label} ({count:,} voxels)")
        
        # Remap: label N -> label N + 180
        data_fixed[right_labels_1_180] = labels_to_remap + 180
        
        print(f"\n  ✓ Remapping complete: 1-180 -> 181-360")
        
        # Verify the remapping
        unique_labels_after = np.unique(data_fixed[data_fixed > 0])
        print(f"\n  After remapping:")
        print(f"    Unique labels: {len(unique_labels_after)}")
        print(f"    Label range: {int(unique_labels_after.min())} to {int(unique_labels_after.max())}")
        
        # Check if we now have the expected 360 labels
        if len(unique_labels_after) == 360 and unique_labels_after.max() == 360:
            print(f"    ✓ Perfect! Now have 360 unique labels (1-360)")
        elif unique_labels_after.max() == 360:
            print(f"    ⚠ Have labels up to 360, but {len(unique_labels_after)} unique labels (expected 360)")
            print(f"      This is normal if some labels are missing in the atlas.")
        
        # Check for remaining overlaps
        left_labels_after = np.unique(data_fixed[left_hemisphere_mask & (data_fixed > 0)])
        right_labels_after = np.unique(data_fixed[right_hemisphere_mask & (data_fixed > 0)])
        overlapping_after = np.intersect1d(left_labels_after, right_labels_after)
        if len(overlapping_after) == 0:
            print(f"    ✓ No overlapping labels after remapping!")
        else:
            print(f"    ⚠ Still have {len(overlapping_after)} overlapping labels after remapping")
    else:
        print("\n✓ No remapping needed. Right hemisphere labels are already correct.")
        unique_labels_after = unique_labels
    
    # Verify against label file if provided (informational only)
    if label_file_path:
        label_file_path = Path(label_file_path)
        if label_file_path.exists():
            print(f"\nNote: Verifying against label file (informational only):")
            print(f"  {label_file_path}")
            print(f"  (Even if labels don't match, remapping is based on spatial location)")
            with open(label_file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Parse label file: "N LabelName" format
            expected_labels = {}
            for line in lines:
                parts = line.split(None, 1)  # Split on whitespace, max 1 split
                if len(parts) == 2:
                    try:
                        label_num = int(parts[0])
                        label_name = parts[1]
                        expected_labels[label_num] = label_name
                    except ValueError:
                        continue
            
            print(f"  Expected labels from file: {len(expected_labels)}")
            print(f"  Actual unique labels in atlas: {len(unique_labels_after)}")
            
            # Convert unique_labels_after to integers for comparison
            unique_labels_after_int = set([int(x) for x in unique_labels_after])
            
            # Check for missing labels
            missing_labels = set(expected_labels.keys()) - unique_labels_after_int
            extra_labels = unique_labels_after_int - set(expected_labels.keys())
            
            if missing_labels:
                print(f"   Missing labels in atlas: {len(missing_labels)}")
                if len(missing_labels) <= 20:
                    print(f"    Missing: {sorted(list(missing_labels))}")
                else:
                    print(f"    Examples: {sorted(list(missing_labels))[:10]}...")
            if extra_labels:
                print(f"   Extra labels in atlas (not in file): {len(extra_labels)}")
                print(f"    Examples: {sorted(list(extra_labels))[:10]}")
            if not missing_labels and not extra_labels:
                print(f"   All labels match!")
    
    # Save the corrected file
    if dry_run:
        print(f"\n[DRY RUN] Would save corrected atlas (but skipping in dry-run mode)")
        if output_path is None:
            # Suggest a default output name
            suggested_output = nifti_path.parent / (nifti_path.stem.replace('.nii', '') + '_FIXED.nii.gz')
            print(f"  Suggested output: {suggested_output}")
        return None
    
    # Default to creating a new file (safer)
    if output_path is None:
        # Create output filename by adding _FIXED before .nii.gz
        if nifti_path.name.endswith('.nii.gz'):
            output_path = nifti_path.parent / (nifti_path.name.replace('.nii.gz', '_FIXED.nii.gz'))
        elif nifti_path.name.endswith('.nii'):
            output_path = nifti_path.parent / (nifti_path.name.replace('.nii', '_FIXED.nii'))
        else:
            output_path = nifti_path.parent / (nifti_path.name + '_FIXED')
        print(f"\nSaving corrected atlas to new file: {output_path}")
    else:
        output_path = Path(output_path)
        print(f"\nSaving corrected atlas to: {output_path}")
    
    # Create new NIfTI image with corrected data
    # Use integer data type for labels (more appropriate and smaller file size)
    if not np.issubdtype(data_fixed.dtype, np.integer):
        data_fixed = data_fixed.astype(np.int32)
    
    # Ensure we're using integer type
    data_fixed = np.round(data_fixed).astype(np.int32)
    data_fixed[data_fixed < 0] = 0
    
    new_img = nib.Nifti1Image(data_fixed, affine, img.header)
    nib.save(new_img, str(output_path))
    
    print(f"  ✓ Saved successfully!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fix HCP-MMP1 atlas labels by remapping right hemisphere from 1-180 to 181-360"
    )
    parser.add_argument(
        "nifti_path",
        type=str,
        help="Path to the input NIfTI atlas file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the corrected NIfTI file (default: creates new file with _FIXED suffix)"
    )
    parser.add_argument(
        "-l", "--label-file",
        type=str,
        default=None,
        help="Path to the label text file for verification (informational only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving (recommended first step)"
    )
    
    args = parser.parse_args()
    
    # If label file not provided, try to find it in the same directory
    if args.label_file is None:
        nifti_path = Path(args.nifti_path)
        label_file_candidate = nifti_path.parent / nifti_path.name.replace('.nii.gz', '.txt').replace('.nii', '.txt')
        if label_file_candidate.exists():
            args.label_file = str(label_file_candidate)
            print(f"Found label file: {args.label_file}")
    
    try:
        output_path = fix_hcp_mmp1_labels(
            args.nifti_path,
            args.output,
            args.label_file,
            dry_run=args.dry_run
        )
        if output_path:
            print(f"\n✓ Done! Corrected atlas saved to: {output_path}")
        elif args.dry_run:
            print(f"\n✓ Dry run complete. Run without --dry-run to apply changes.")
        else:
            print(f"\n✓ Done! (No changes were needed or operation was cancelled)")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

