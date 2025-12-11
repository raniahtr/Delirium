#!/usr/bin/env python3
"""
Generate a simple codebook for the reindexed 430-label atlas.

Input atlas:
  /media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Final_Combined_atlas_MNI2009c_1mm_reindexed.nii.gz

Output:
  /media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/codebook_430.mat

Fields saved:
  CodeBook.full   : cellstr-equivalent (object array) of long names ("Label_<id>")
  CodeBook.short  : cellstr-equivalent of short names ("L<id>")
  CodeBook.xyz    : (430 x 3) centroid coordinates in MNI space
"""

import numpy as np
import nibabel as nib
import scipy.io as sio

ATLAS_PATH = "/media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Final_Combined_atlas_MNI2009c_1mm_reindexed.nii.gz"
OUT_PATH = "/media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/codebook_430.mat"


def main():
    img = nib.load(ATLAS_PATH)
    data = img.get_fdata()
    affine = img.affine

    labels = np.unique(data[data > 0]).astype(int)
    labels = np.sort(labels)

    xyz = []
    full = []
    short = []
    for lbl in labels:
        ijk = np.argwhere(data == lbl)
        if ijk.size == 0:
            xyz.append([np.nan, np.nan, np.nan])
        else:
            mm = nib.affines.apply_affine(affine, ijk)
            xyz.append(mm.mean(axis=0))
        full.append(f"Label_{lbl}")
        short.append(f"L{lbl}")

    xyz_arr = np.array(xyz, dtype=float)
    full_arr = np.array(full, dtype=object).reshape(-1, 1)
    short_arr = np.array(short, dtype=object).reshape(-1, 1)
    id_arr = np.arange(1, len(labels) + 1, dtype=np.int32).reshape(-1, 1)

    # Build fields expected by show_cm_extended
    center_cells = np.empty((1, len(labels)), dtype=object)
    for i, coord in enumerate(xyz_arr):
        center_cells[0, i] = coord.reshape(3, 1)

    # Build MATLAB-style struct
    CodeBook = np.zeros(
        1,
        dtype=[
            ("center", "O"),
            ("id", "O"),
            ("name", "O"),
            ("sname", "O"),
            ("num", "O"),
        ],
    )
    CodeBook["center"][0] = center_cells
    CodeBook["id"][0] = id_arr
    CodeBook["name"][0] = full_arr
    CodeBook["sname"][0] = short_arr
    CodeBook["num"][0] = np.array([[len(labels)]], dtype=np.int32)

    sio.savemat(OUT_PATH, {"CodeBook": CodeBook}, do_compression=True)
    print(f"Saved codebook with {len(labels)} entries to {OUT_PATH}")
    print(f"Label range: {labels.min()}..{labels.max()}")


if __name__ == "__main__":
    main()

