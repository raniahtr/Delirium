#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_new_clean"

cd "$ROOT"

for sub in sub-*; do
  echo "=============================="
  echo "Processing $sub"
  echo "=============================="

  cd "$sub"

  # ---------- STEP 1: Deep white-matter mask from 5tt_coreg.mif ----------
  if [ -f 5tt_coreg.mif ] && [ -f mean_b0.nii.gz ]; then
    echo "[WM] Extracting WM prob from 5tt_coreg.mif..."

    # 1) extract WM probability map (component index 2)
    mrconvert 5tt_coreg.mif -coord 3 2 wm_prob.mif -force

    # 2) threshold to keep only high-probability WM voxels
    mrthreshold wm_prob.mif -abs 0.7 wm_mask.mif -force

    # 3) erode to get "deep" WM, away from borders
    maskfilter wm_mask.mif erode -npass 2 wm_deep.mif -force

    # 4) ***RESAMPLE deep WM mask to DWI/b0 space***
    # use nearest-neighbour to preserve binary mask
    mrtransform wm_deep.mif \
      -template mean_b0.nii.gz \
      -interp nearest \
      wm_deep_dwi.mif

    # 5) convert to NIfTI for Python
    mrconvert wm_deep_dwi.mif wm_deep_dwi.nii.gz -force

  else
    echo "[WM] WARNING: 5tt_coreg.mif or mean_b0.nii.gz not found in $sub – skipping WM mask."
  fi

  # ---------- STEP 2: Convert noise.mif to NIfTI ----------
  if [ -f noise.mif ]; then
    echo "[NOISE] Converting noise.mif → noise.nii.gz..."
    mrconvert noise.mif noise.nii.gz -force
  else
    echo "[NOISE] WARNING: noise.mif not found in $sub – skipping noise conversion."
  fi

  cd ..
done

echo "Done: deep WM masks in DWI space (wm_deep_dwi.nii.gz) and noise.nii.gz created for all subjects."
