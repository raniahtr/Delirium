#!/usr/bin/env python3
"""
Create a reindexed atlas NIfTI that removes specific labels and shifts the remaining
labels down to keep the index order contiguous.

Source atlas:
  /media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Final_Combined_atlas_MNI2009c_1mm.nii.gz

Removed labels (1-indexed):
  390, 423   (these were dropped from the connectomes)

Output:
  /media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Final_Combined_atlas_MNI2009c_1mm_reindexed.nii.gz
"""

import numpy as np
import nibabel as nib
from pathlib import Path

SRC = Path("/media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Final_Combined_atlas_MNI2009c_1mm.nii.gz")
DST = Path("/media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Final_Combined_atlas_MNI2009c_1mm_reindexed.nii.gz")

REMOVED = {390, 423}


def main():
    img = nib.load(str(SRC))
    data = img.get_fdata()
    data_int = data.astype(np.int32)

    labels = np.unique(data_int[data_int > 0])
    kept = [int(l) for l in labels if l not in REMOVED]
    kept_sorted = sorted(kept)

    print(f"Found {len(labels)} labels >0; keeping {len(kept_sorted)}, removing {len(REMOVED)}")

    # Build lookup table mapping old_label -> new_label (0 stays 0)
    lut_size = int(labels.max()) + 1
    lut = np.zeros(lut_size, dtype=np.int32)
    new_id = 1
    for old in kept_sorted:
        lut[old] = new_id
        new_id += 1

    # Apply mapping
    mapped = lut[data_int]

    # Save
    out_img = nib.Nifti1Image(mapped.astype(np.int32), img.affine, img.header)
    nib.save(out_img, str(DST))

    print(f"Saved reindexed atlas to: {DST}")
    print(f"Max new label: {mapped.max()}")


if __name__ == "__main__":
    main()

