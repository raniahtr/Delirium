#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG (ABSOLUTE PATHS) ==================
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
PREPROC_ROOT="${ROOT}/Preproc"
ATLAS_3D="${ROOT}/atlas/HCPMMP1_labels_3D.nii.gz"     # 3D integer Glasser labels
FSLDIR_DEFAULT="/usr/local/fsl"
MNI_T1="${FSLDIR:-$FSLDIR_DEFAULT}/data/standard/MNI152_T1_1mm.nii.gz"
ASSIGN_RADIAL=3   # slightly more tolerant than 2
# =============================================================

log(){ echo -e "\n[Batch] $*"; }

[[ -f "$ATLAS_3D" ]] || { echo "Atlas not found: $ATLAS_3D"; exit 1; }
[[ -f "$MNI_T1"   ]] || { echo "MNI T1 not found: $MNI_T1"; exit 1; }
[[ -d "$PREPROC_ROOT" ]] || { echo "Preproc folder not found: $PREPROC_ROOT"; exit 1; }

cd "$PREPROC_ROOT"

for sdir in sub-*; do
  [[ -d "$sdir" ]] || continue
  sid="${sdir#sub-}"
  log "Processing $sdir"

  cd "$sdir"

  # Ensure we can write results
  mkdir -p connectome_local
  chmod -R u+rwX connectome_local || true

  # Required inputs
  if [[ ! -f "mean_b0.mif" || ! -f "diff2struct_mrtrix.txt" ]]; then
    echo "[Skip] Missing mean_b0.mif or diff2struct_mrtrix.txt in $sdir"
    cd ..; continue
  fi

  # T1: prefer T1.mif; else convert first *_T1w.nii.gz
  if [[ -f "T1.mif" ]]; then
    mrconvert T1.mif T1.nii.gz -quiet -force
  elif ls *_T1w.nii.gz >/dev/null 2>&1; then
    t1src="$(ls *_T1w.nii.gz | head -n1)"
    mrconvert "$t1src" T1.nii.gz -quiet -force
  else
    echo "[Skip] No T1 found in $sdir"
    cd ..; continue
  fi

  # 1) MNI -> T1 (intensity registration)
  flirt -in "$MNI_T1" -ref T1.nii.gz -omat MNItoT1.mat -dof 12 -cost corratio

  # 2) Apply Glasser labels to T1 (NN)
  flirt -in "$ATLAS_3D" -ref T1.nii.gz \
        -applyxfm -init MNItoT1.mat -interp nearestneighbour \
        -out atlas_in_T1.nii.gz

  mrconvert atlas_in_T1.nii.gz atlas_in_T1.mif -quiet -force

  # 3) Invert DWI->T1 to T1->DWI
  transformcalc diff2struct_mrtrix.txt invert struct2diff.txt -force

  # 4) Push labels to DWI space (NN, template mean_b0)
  mrtransform atlas_in_T1.mif -linear struct2diff.txt \
              -template mean_b0.mif -interp nearest \
              atlas_in_dwi.mif -quiet -force

  # 4b) Convert to integer labels to silence float warning
  mrcalc atlas_in_dwi.mif 0.5 -add -floor -datatype uint16 atlas_in_dwi_uint16.mif -force

  # QC: check atlas range
  minv=$(mrstats atlas_in_dwi_uint16.mif -output min)
  maxv=$(mrstats atlas_in_dwi_uint16.mif -output max)
  echo "[QC] atlas_in_dwi_uint16 min/max: $minv / $maxv"

  # 5) Build connectome (prefer SIFT2; fallback to counts)
  tck="sub-${sid}_filtered.tck"
  if [[ ! -f "$tck" ]]; then
    if ls *_filtered.tck >/dev/null 2>&1; then
      tck="$(ls *_filtered.tck | head -n1)"
    else
      echo "[Skip] No filtered .tck found in $sdir"
      cd ..; continue
    fi
  fi

  out_sift="connectome_local/sub-${sid}_SC_sift2.csv"
  out_cnts="connectome_local/sub-${sid}_SC_counts.csv"
  wts="sub-${sid}_sift_coeffs.txt"
  if [[ ! -f "$wts" && -n "$(ls sub-*_sift_coeffs.txt 2>/dev/null | head -n1)" ]]; then
    wts="$(ls sub-*_sift_coeffs.txt | head -n1)"
  fi

  used=""
  if [[ -f "$wts" ]]; then
    tck2connectome "$tck" atlas_in_dwi_uint16.mif "$out_sift" \
      -tck_weights_in "$wts" -symmetric -zero_diagonal \
      -assignment_radial_search "$ASSIGN_RADIAL" -force
    used="$out_sift"
  else
    echo "[Info] No SIFT2 weights for $sid → using counts."
    tck2connectome "$tck" atlas_in_dwi_uint16.mif "$out_cnts" \
      -symmetric -zero_diagonal \
      -assignment_radial_search "$ASSIGN_RADIAL" -force
    used="$out_cnts"
  fi

  # Guard: make sure the file exists before QC
  if [[ ! -f "$used" ]]; then
    echo "[ERR] Connectome not written for $sid"; cd ..; continue
  fi

  # Tiny QC with robust Python (no argv issues)
  python3 - "$used" <<'PY'
import sys, numpy as np
p = sys.argv[1]
M = np.loadtxt(p, delimiter=",")
nnz = int((M!=0).sum())
empties = np.where((M.sum(0)+M.sum(1))==0)[0]
print(f"[QC] {p} non-zero entries: {nnz}")
if len(empties):
    print(f"[QC] Empty nodes (indices): {empties.tolist()}")
PY

  cd ..
done

echo -e "\n[Batch] Done."
