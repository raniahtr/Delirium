#!/usr/bin/env bash
set -euo pipefail

# ---- Paths ----
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
PREPROC="$ROOT/Preproc_new_clean"
ATLAS_GLR="$ROOT/atlas/HCPMMP1_labels_3D.nii.gz"          # Glasser (1mm)
HCPEX_DIR="$ROOT/atlas/HCPex_folder"

# Accept any of these; we’ll normalize below
HCPEX_CANDIDATES=("HCPex_2mm.nii" "HCPex_2mm.nii.gz" "HCPex.nii.gz" "HCPex.nii")
HCPEX_2MM=""     # will be set after detection
HCPEX_1MM="$HCPEX_DIR/HCPex_1mm_resampled.nii.gz"

MNI_T1=${FSLDIR:-/usr/local/fsl}/data/standard/MNI152_T1_1mm.nii.gz

# ---- Detect HCPex file name gracefully ----
for cand in "${HCPEX_CANDIDATES[@]}"; do
  if [[ -f "$HCPEX_DIR/$cand" ]]; then
    HCPEX_2MM="$HCPEX_DIR/$cand"
    break
  fi
done
if [[ -z "$HCPEX_2MM" ]]; then
  echo "[ERROR] No HCPex volumetric file found in $HCPEX_DIR"
  echo "        Expected one of: ${HCPEX_CANDIDATES[*]}"
  exit 1
fi
echo "[Atlas] Using HCPex source: $(basename "$HCPEX_2MM")"

# ---- Ensure HCPex 1mm exists (resample once) ----
if [[ ! -f "$HCPEX_1MM" ]]; then
  echo "[Prep] Resampling HCPex (→ 1mm) to match Glasser..."
  flirt -in "$HCPEX_2MM" \
        -ref "$ATLAS_GLR" \
        -applyisoxfm 1 \
        -interp nearestneighbour \
        -out "$HCPEX_1MM"
  echo "[Prep] Created: $HCPEX_1MM"
fi

process_subject () {
  local SUBJ="$1"
  local SDIR="$PREPROC/$SUBJ"
  [[ -d "$SDIR" ]] || { echo "[Skip] $SUBJ: no dir"; return; }
  cd "$SDIR"

  echo -e "\n=== $SUBJ ==="

  # Required inputs
  if [[ ! -f mean_b0.mif || ! -f diff2struct_mrtrix.txt ]]; then
    echo "[Skip] Missing mean_b0.mif or diff2struct_mrtrix.txt"
    cd - >/dev/null; return
  fi
  [[ -f T1.nii.gz ]] || { [[ -f T1.mif ]] && mrconvert T1.mif T1.nii.gz -force -quiet; }
  [[ -f T1.nii.gz ]] || { echo "[Skip] $SUBJ: no T1"; cd - >/dev/null; return; }

  # ---------- MNI -> T1 (affine) ----------
  echo "[Warp] Estimating MNI→T1 affine"
  flirt -in "$MNI_T1" -ref T1.nii.gz -omat MNItoT1.mat -dof 12 -cost corratio

  # ---------- Glasser (1–360) ----------
  if [[ ! -f atlas_in_dwi_uint16.mif ]]; then
    echo "[Map] Glasser MMP1 to T1"
    flirt -in "$ATLAS_GLR" -ref T1.nii.gz -applyxfm -init MNItoT1.mat \
          -interp nearestneighbour -out glasser_in_T1.nii.gz

    echo "[Map] Glasser T1→DWI"
    mrconvert glasser_in_T1.nii.gz glasser_in_T1.mif -quiet -force
    transformcalc diff2struct_mrtrix.txt invert struct2diff.txt -force
    mrtransform glasser_in_T1.mif -linear struct2diff.txt \
                -template mean_b0.mif -interp nearest glasser_in_dwi.mif -quiet -force
    # Clean uint16 (1–360): round to nearest integer, preserving labels 1-360
    # Ensure non-negative values first (handle any precision artifacts), then round
    # Standard rounding: floor(x + 0.5) preserves integers and correctly rounds near-integers
    mrcalc glasser_in_dwi.mif 0 -max glasser_nonneg.mif -quiet -force
    mrcalc glasser_nonneg.mif 0.5 -add -floor -datatype uint16 atlas_in_dwi_uint16.mif -quiet -force
    rm -f glasser_nonneg.mif
  else
    echo "[Reuse] Found Glasser in DWI: atlas_in_dwi_uint16.mif"
    transformcalc diff2struct_mrtrix.txt invert struct2diff.txt -force  # ensure exists
  fi

  # ---------- HCPex (keep only >360) ----------
  echo "[Map] HCPex (1mm) to T1"
  flirt -in "$HCPEX_1MM" -ref T1.nii.gz -applyxfm -init MNItoT1.mat \
        -interp nearestneighbour -out hcpex_in_T1.nii.gz

  echo "[Map] HCPex T1→DWI"
  mrconvert hcpex_in_T1.nii.gz hcpex_in_T1.mif -quiet -force
  mrtransform hcpex_in_T1.mif -linear struct2diff.txt \
              -template mean_b0.mif -interp nearest hcpex_in_dwi.mif -quiet -force

  # ints, then mask >360
  mrcalc hcpex_in_dwi.mif 0.5 -add -floor -datatype uint16 hcpex_uint16.mif -quiet -force
  mrcalc hcpex_uint16.mif 360 -gt hcpex_sub_mask.mif -quiet -force
  mrcalc hcpex_uint16.mif hcpex_sub_mask.mif -mult hcpex_sub_only.mif -quiet -force

  # ---------- Merge ----------
  mrcalc atlas_in_dwi_uint16.mif hcpex_sub_only.mif -max Combined_HCPex_426.mif -quiet -force
  mrconvert Combined_HCPex_426.mif Combined_HCPex_426.nii.gz -quiet -force

  # ---------- QC ----------
  read gmin gmax <<<"$(mrstats atlas_in_dwi_uint16.mif -output min -output max)"
  read smin smax <<<"$(mrstats hcpex_sub_only.mif -output min -output max)"
  read cmin cmax <<<"$(fslstats Combined_HCPex_426.nii.gz -R)"
  echo "[QC] Glasser_in_DWI   (expect 1–360): min=$gmin max=$gmax"
  echo "[QC] HCPex_sub_only   (expect 361–426): min=$smin max=$smax"
  echo "[QC] Combined         (expect 1–426): min=$cmin max=$cmax"

  # Optional connectome
  if [[ -f tracks_10M.tck ]]; then
    echo "[Connectome] tck2connectome → connectome_426.csv"
    tck2connectome tracks_10M.tck Combined_HCPex_426.mif connectome_426.csv \
      -assignment_radial_search 2 -force -quiet
  fi

  # Cleanup
  rm -f glasser_in_T1.nii.gz glasser_in_T1.mif hcpex_in_T1.nii.gz hcpex_in_T1.mif \
        struct2diff.txt MNItoT1.mat
  cd - >/dev/null
}

# ------------------ DRIVER ------------------
if [[ "${1:-}" == "all" ]]; then
  cd "$PREPROC"
  for s in sub-*; do [[ -d "$s" ]] && process_subject "$s"; done
elif [[ -n "${1:-}" ]]; then
  process_subject "$1"
else
  echo "Usage: bash $(basename "$0") sub-XX | all"
  exit 1
fi
echo "[Batch] Done."
