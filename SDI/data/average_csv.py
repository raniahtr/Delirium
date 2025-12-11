#!/usr/bin/env python3
"""
Average SIFT2 size-corrected structural connectomes into a group mean .mat file.

Inputs:
  - CSV files: /media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_current/sub-*/connectome_local/*sift2_sizecorr*.csv
Outputs:
  - MAT file:  /media/RCPNAS/Data/Delirium/Delirium_Rania/SDI/data/SC_A.mat
"""

import glob
import os
import numpy as np
import scipy.io as sio

# Absolute base directory for the project
BASE_DIR = "/media/RCPNAS/Data/Delirium/Delirium_Rania"

# Pattern to find all SIFT2 size-corrected connectome CSVs
CSV_GLOB = os.path.join(BASE_DIR, "Preproc_current/sub-*/connectome_local/*sift2_sizecorr*.csv")

# Output .mat path
OUT_PATH = os.path.join(BASE_DIR, "SDI/data/SC_A.mat")

def main():
    # Locate CSVs
    csv_paths = sorted(glob.glob(CSV_GLOB))
    if not csv_paths:
        raise SystemExit(f"No SIFT2 size-corrected CSVs found with pattern: {CSV_GLOB}")

    # Load all matrices
    mats = []
    for path in csv_paths:
        arr = np.loadtxt(path, delimiter=",")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mats.append(arr)

    # Stack and average
    stack = np.stack(mats, axis=0)           # shape: (n_subjects, n_nodes, n_nodes)
    mean_mat = stack.mean(axis=0)            # group average

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Save to .mat with a clear variable name
    sio.savemat(OUT_PATH, {"A_mean": mean_mat})

    print(f"Saved group-average matrix to {OUT_PATH} with shape {mean_mat.shape}")
    print(f"Used {len(csv_paths)} input file(s).")

if __name__ == "__main__":
    main()