#!/usr/bin/env python3
"""
Average SIFT2 size-corrected structural connectomes into mean .mat files.

Outputs:
- Whole cohort: SDI/data/SC_A.mat
- Groups:
  * SDI/data/SC_A_ICU.mat
  * SDI/data/SC_A_ICU_delirium.mat
  * SDI/data/SC_A_HC.mat
  * SDI/data/SC_A_ICU_combined.mat (ICU + ICU_delirium combined)
"""

import glob
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

BASE_DIR = Path("/media/RCPNAS/Data/Delirium/Delirium_Rania")

# Pattern to find all SIFT2 size-corrected connectome CSVs
CSV_GLOB = str(BASE_DIR / "Preproc_current" / "sub-*" / "connectome_local" / "*sift2_sizecorr*.csv")

OUT_DIR = BASE_DIR / "SDI" / "data"
OUT_WHOLE = OUT_DIR / "SC_A.mat"

# Group definitions (subject codes without 'sub-' prefix)
GROUPS = {
    "ICU": ["AF", "DA2", "PM", "BA", "VC"],
    "ICU_delirium": ["CG", "DA", "FS", "FSE", "GL", "KJ", "LL", "MF", "PMA", "PO", "PB", "SA"],
    "HC": ["FEF", "FD", "GB", "SG", "AR", "TL", "TOG", "PL", "ZM", "AM", "PC", "AD"],
}


def load_matrices(csv_paths):
    mats = []
    subj_ids = []
    for path in csv_paths:
        arr = np.loadtxt(path, delimiter=",")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mats.append(arr)
        fname = os.path.basename(path)
        subj_prefix = fname.split("_")[0]  # sub-XXX
        subj_ids.append(subj_prefix.replace("sub-", "").upper())
    return np.stack(mats, axis=0), np.array(subj_ids)


def save_mean(mat_mean, out_path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_path, {"A_mean": mat_mean})
    print(f"Saved: {out_path} shape={mat_mean.shape}")


def main():
    # Locate CSVs
    csv_paths = sorted(glob.glob(CSV_GLOB))
    if not csv_paths:
        raise SystemExit(f"No SIFT2 size-corrected CSVs found with pattern: {CSV_GLOB}")

    stack, subj_ids = load_matrices(csv_paths)

    # Whole cohort
    mean_mat = stack.mean(axis=0)
    save_mean(mean_mat, OUT_WHOLE)
    print(f"Used {len(csv_paths)} input file(s) for whole cohort.")

    # Groups
    for group_name, codes in GROUPS.items():
        mask = np.isin(subj_ids, [c.upper() for c in codes])
        if not mask.any():
            print(f"Warning: no subjects found for group '{group_name}'")
            continue
        group_stack = stack[mask]
        group_mean = group_stack.mean(axis=0)
        out_path = OUT_DIR / f"SC_A_{group_name}.mat"
        save_mean(group_mean, out_path)
        print(f"Group '{group_name}': used {group_stack.shape[0]} subject(s)")
    
    # Combined ICU + ICU_delirium
    icu_codes = [c.upper() for c in GROUPS["ICU"]]
    icu_delirium_codes = [c.upper() for c in GROUPS["ICU_delirium"]]
    combined_codes = icu_codes + icu_delirium_codes
    combined_mask = np.isin(subj_ids, combined_codes)
    
    if combined_mask.any():
        combined_stack = stack[combined_mask]
        combined_mean = combined_stack.mean(axis=0)
        out_path = OUT_DIR / "SC_A_ICU_combined.mat"
        save_mean(combined_mean, out_path)
        print(f"Combined ICU group: used {combined_stack.shape[0]} subject(s)")
    else:
        print("Warning: no subjects found for combined ICU group")


if __name__ == "__main__":
    main()