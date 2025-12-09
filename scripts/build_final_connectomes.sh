#!/usr/bin/env bash
# build_final_connectomes.sh
# Creates connectomes for the Final_Combined_atlas_MNI2009c_1mm (432 parcels) per subject.
# The atlas is in MNI2009c space and is registered to each subject's DWI space.
#
# Outputs (in connectome_local/):
#   - ${SUBJ}_SC_sift2.csv          (sum of SIFT2 weights; falls back to counts if no weights)
#   - ${SUBJ}_SC_invlen_sum.csv     (sum of inverse lengths; UNWEIGHTED)
#   - ${SUBJ}_COUNT.csv             (streamline count; UNWEIGHTED)
#   - ${SUBJ}_TOTAL_length.csv      (total streamline length; UNWEIGHTED)
#   - ${SUBJ}_MEAN_length.csv       (mean streamline length; UNWEIGHTED)
#   - ${SUBJ}_MEAN_FA.csv           (edge mean of per-stream FA; SIFT2-weighted if weights exist)
#   - ${SUBJ}_MEAN_AD.csv
#   - ${SUBJ}_MEAN_RD.csv
#   - ${SUBJ}_MEAN_MD.csv
#   - ${SUBJ}_FA_perstream.txt, ${SUBJ}_AD_perstream.txt, ${SUBJ}_RD_perstream.txt, ${SUBJ}_MD_perstream.txt
#
# Usage:
#   bash build_final_connectomes.sh sub-XX
#   bash build_final_connectomes.sh all
#
# Assumptions:
#   - Working tree: ROOT/sub-XX/ with:
#       mean_b0.mif (or mean_b0.nii.gz)
#       diff2struct_mrtrix.txt (DWI→T1 transform)
#       T1.mif or sub-*_T1w.nii.gz (T1 anatomical)
#       ${SUBJ}_tracks_10M.tck or ${SUBJ}_filtered.tck
#       (optional) ${SUBJ}_sift_1M.txt or similar SIFT weights
#       (optional) ${SUBJ}_{FA,MD}.{mif|nii|nii.gz}
#   - MRtrix3 and FSL on PATH
#   - Single atlas: Final_Combined_atlas_MNI2009c_1mm.nii.gz (432 labels) in atlas/
#
set -euo pipefail

# --------- CONFIG ---------
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
PREPROC_ROOT="${ROOT}/Preproc_current"
ATLAS_MNI="${ROOT}/atlas/Final_Combined_atlas_MNI2009c_1mm.nii.gz"
MNI_T1="${FSLDIR:-/usr/local/fsl}/data/standard/MNI152_T1_1mm.nii.gz"

# Tractogram name patterns (will try in order)
TCK_PATTERNS=(
  "_tracks_10M.tck"
  "_filtered.tck"
  "_tracks.tck"
)

# tck->atlas assignment options
ASSIGN_OPTS=(-assignment_radial_search 3 -symmetric -zero_diagonal)

# Expected max label in atlas
EXPECTED_MAX_LABEL=432

usage(){ echo "Usage: $(basename "$0") sub-XX | all"; exit 1; }
[[ $# -eq 1 ]] || usage

# --------- HELPERS ---------

# Accepts base image path without extension, returns the first found path
# priority: .mif > .nii.gz > .nii
find_image(){
  local base="$1"
  if [[ -f "${base}.mif" ]]; then
    echo "${base}.mif"
  elif [[ -f "${base}.nii.gz" ]]; then
    echo "${base}.nii.gz"
  elif [[ -f "${base}.nii" ]]; then
    echo "${base}.nii"
  else
    echo ""
  fi
}

# Find tractogram file for subject
find_tractogram(){
  local SUBJ="$1"
  local SDIR="$2"
  for pattern in "${TCK_PATTERNS[@]}"; do
    local candidate="${SDIR}/${SUBJ}${pattern}"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo ""
}

# Weighted tck2connectome (uses -tck_weights_in when TCKW set)
run_t2c_weighted(){
  local TCK="$1" ATLAS="$2" OUTCSV="$3"; shift 3
  if [[ -n "${TCKW:-}" ]]; then
    tck2connectome "$TCK" "$ATLAS" "$OUTCSV" \
      -tck_weights_in "$TCKW" "${ASSIGN_OPTS[@]}" -force "$@"
  else
    # Fall back to unweighted if weights missing (warn upstream if that matters)
    tck2connectome "$TCK" "$ATLAS" "$OUTCSV" \
      "${ASSIGN_OPTS[@]}" -force "$@"
  fi
}

# Unweighted tck2connectome (explicitly ignores TCKW)
run_t2c_unweighted(){
  local TCK="$1" ATLAS="$2" OUTCSV="$3"; shift 3
  tck2connectome "$TCK" "$ATLAS" "$OUTCSV" \
    "${ASSIGN_OPTS[@]}" -force "$@"
}

# Make MEAN_<TAG> by sampling per-streamline values then aggregating edges (weighted if TCKW present)
make_metric_connectome(){
  local SUBJ="$1" TCK="$2" ATLAS="$3" IMG="$4" TAG="$5" OUTDIR="$6"
  local SCALES="$OUTDIR/${SUBJ}_${TAG}_perstream.txt"
  echo "[Info] Sampling ${TAG} per-streamline -> $SCALES"
  tcksample "$TCK" "$IMG" "$SCALES" -stat_tck mean -force -quiet
  local OUTCSV="$OUTDIR/${SUBJ}_MEAN_${TAG}.csv"
  echo "[Build] MEAN_${TAG} -> $OUTCSV"
  run_t2c_weighted "$TCK" "$ATLAS" "$OUTCSV" -scale_file "$SCALES" -stat_edge mean
}

# Register MNI atlas to subject DWI space
# Returns path to atlas in DWI space (uint16 .mif)
register_atlas_to_dwi(){
  local SUBJ="$1" SDIR="$2" OUTDIR="$3"
  
  # Check required files
  local MEAN_B0 MEAN_B0_NII T1 T1_NII DIFF2STRUCT
  MEAN_B0="$(find_image "${SDIR}/mean_b0")"
  [[ -n "$MEAN_B0" ]] || { echo "[ERROR] mean_b0.{mif|nii|nii.gz} not found" >&2; return 1; }
  
  # Convert mean_b0 to .nii.gz if needed for FSL
  if [[ "$MEAN_B0" == *.mif ]]; then
    MEAN_B0_NII="${SDIR}/mean_b0.nii.gz"
    [[ -f "$MEAN_B0_NII" ]] || mrconvert "$MEAN_B0" "$MEAN_B0_NII" -force -quiet
  else
    MEAN_B0_NII="$MEAN_B0"
  fi
  
  # Find T1
  T1="$(find_image "${SDIR}/T1")"
  if [[ -z "$T1" ]]; then
    # Try subject-specific T1 naming
    T1="$(find_image "${SDIR}/${SUBJ}_T1w")"
  fi
  [[ -n "$T1" ]] || { echo "[ERROR] T1.{mif|nii|nii.gz} or ${SUBJ}_T1w.{mif|nii|nii.gz} not found" >&2; return 1; }
  
  # Convert T1 to .nii.gz if needed for FSL
  if [[ "$T1" == *.mif ]]; then
    T1_NII="${SDIR}/T1.nii.gz"
    [[ -f "$T1_NII" ]] || mrconvert "$T1" "$T1_NII" -force -quiet
  else
    T1_NII="$T1"
  fi
  
  # Check for DWI→T1 transform
  DIFF2STRUCT="${SDIR}/diff2struct_mrtrix.txt"
  [[ -f "$DIFF2STRUCT" ]] || { echo "[ERROR] diff2struct_mrtrix.txt not found" >&2; return 1; }
  
  echo "[Register] Registering MNI atlas to ${SUBJ} DWI space..." >&2
  
  # Step 1: MNI T1 → Subject T1 (intensity-based registration)
  local MNI2T1_MAT="${OUTDIR}/MNI2T1.mat"
  if [[ ! -f "$MNI2T1_MAT" ]]; then
    echo "[Register] Step 1/3: MNI T1 → Subject T1 (FLIRT)" >&2
    flirt -in "$MNI_T1" -ref "$T1_NII" -omat "$MNI2T1_MAT" \
      -dof 12 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 \
      -interp trilinear
  else
    echo "[Register] Reusing existing MNI2T1.mat" >&2
  fi
  
  # Step 2: Apply transform to atlas (MNI → T1 space) with nearest-neighbor
  local ATLAS_IN_T1="${OUTDIR}/atlas_in_T1.nii.gz"
  if [[ ! -f "$ATLAS_IN_T1" ]]; then
    echo "[Register] Step 2/3: Applying transform to atlas (MNI → T1, nearest-neighbor)" >&2
    flirt -in "$ATLAS_MNI" -ref "$T1_NII" \
      -applyxfm -init "$MNI2T1_MAT" \
      -interp nearestneighbour \
      -out "$ATLAS_IN_T1"
  else
    echo "[Register] Reusing existing atlas_in_T1.nii.gz" >&2
  fi
  
  # Convert to .mif for MRtrix
  local ATLAS_IN_T1_MIF="${OUTDIR}/atlas_in_T1.mif"
  [[ -f "$ATLAS_IN_T1_MIF" ]] || mrconvert "$ATLAS_IN_T1" "$ATLAS_IN_T1_MIF" -force -quiet
  
  # Step 3: T1 → DWI (invert diff2struct transform)
  local STRUCT2DIFF="${OUTDIR}/struct2diff.txt"
  if [[ ! -f "$STRUCT2DIFF" ]]; then
    echo "[Register] Step 3/3: Inverting DWI→T1 transform (T1 → DWI)" >&2
    transformcalc "$DIFF2STRUCT" invert "$STRUCT2DIFF" -force
  else
    echo "[Register] Reusing existing struct2diff.txt" >&2
  fi
  
  # Apply T1→DWI transform, regrid to mean_b0, nearest-neighbor interpolation
  local ATLAS_IN_DWI="${OUTDIR}/atlas_in_dwi_uint16.mif"
  if [[ ! -f "$ATLAS_IN_DWI" ]]; then
    echo "[Register] Transforming atlas to DWI space (T1 → DWI, nearest-neighbor)" >&2
    mrtransform "$ATLAS_IN_T1_MIF" \
      -linear "$STRUCT2DIFF" \
      -template "$MEAN_B0" \
      -interp nearest \
      "$ATLAS_IN_DWI" \
      -force -quiet
    
    # Ensure uint16 datatype
    local ATLAS_CHECK
    ATLAS_CHECK=$(mrinfo "$ATLAS_IN_DWI" -datatype)
    if [[ "$ATLAS_CHECK" != "uint16" ]]; then
      echo "[Convert] Converting atlas to uint16..." >&2
      local ATLAS_TMP="${OUTDIR}/atlas_in_dwi_tmp.mif"
      mrconvert "$ATLAS_IN_DWI" "$ATLAS_TMP" -datatype uint16 -force -quiet
      mv "$ATLAS_TMP" "$ATLAS_IN_DWI"
    fi
  else
    echo "[Register] Reusing existing atlas_in_dwi_uint16.mif" >&2
  fi
  
  # QC: Verify label range
  read -r MIN MAX <<<"$(mrstats "$ATLAS_IN_DWI" -output min -output max)"
  echo "[QC] Atlas in DWI space: min=$MIN max=$MAX (expect max=$EXPECTED_MAX_LABEL)" >&2
  if [[ "$MAX" != "$EXPECTED_MAX_LABEL" ]]; then
    echo "[WARNING] Max label ($MAX) does not match expected ($EXPECTED_MAX_LABEL)" >&2
    echo "          This may indicate a registration issue, but continuing..." >&2
  fi
  
  # Only echo the file path to stdout (for command substitution)
  echo "$ATLAS_IN_DWI"
}

build_one(){
  local SUBJ="$1"
  local SDIR="$PREPROC_ROOT/$SUBJ"
  echo "=== $SUBJ ==="
  [[ -d "$SDIR" ]] || { echo "[ERROR] $SDIR not found"; return 1; }
  cd "$SDIR"
  
  local OUTDIR="connectome_local"
  mkdir -p "$OUTDIR"
  
  # Register atlas to DWI space
  local ATLAS
  ATLAS="$(register_atlas_to_dwi "$SUBJ" "$SDIR" "$OUTDIR")" || return 1
  
  # Find tractogram
  local TCK
  TCK="$(find_tractogram "$SUBJ" "$SDIR")"
  [[ -n "$TCK" ]] || { echo "[ERROR] No tractogram found (tried: ${TCK_PATTERNS[*]})"; return 1; }
  echo "[Info] Using tractogram: $(basename "$TCK")"
  
  # Find SIFT2 weights (optional, but will upgrade weighted ops when present)
  # preprocessing.sh creates: ${SUBJ}_sift_1M.txt (weights), ${SUBJ}_sift_mu.txt (parameter), ${SUBJ}_sift_coeffs.txt (coefficients)
  # Only the _sift_1M.txt file contains the actual weights for tck2connectome
  TCKW=""
  if [[ -f "${SUBJ}_sift_1M.txt" ]]; then
    TCKW="${SUBJ}_sift_1M.txt"
  fi
  
  if [[ -n "$TCKW" ]]; then
    echo "[Info] Using SIFT weights: $TCKW"
  else
    echo "[Warn] No SIFT weights found — SC_sift2 will fall back to counts; MEAN_* will be unweighted means."
  fi
  
  # --------- (1) SC_sift2 (sum of weights; or counts if no weights) ---------
  local SC_SIFT2="$OUTDIR/${SUBJ}_SC_sift2.csv"
  echo "[Build] SC_sift2 -> $SC_SIFT2"
  run_t2c_weighted "$TCK" "$ATLAS" "$SC_SIFT2" -stat_edge sum
  
  # --------- (2) SC_invlen_sum (UNWEIGHTED) ---------
  local SC_INVLEN="$OUTDIR/${SUBJ}_SC_invlen_sum.csv"
  echo "[Build] SC_invlen_sum (unweighted) -> $SC_INVLEN"
  run_t2c_unweighted "$TCK" "$ATLAS" "$SC_INVLEN" -scale_invlength -stat_edge sum
  
  # --------- (3) COUNT (UNWEIGHTED) ---------
  local COUNT_CSV="$OUTDIR/${SUBJ}_COUNT.csv"
  echo "[Build] COUNT (unweighted) -> $COUNT_CSV"
  run_t2c_unweighted "$TCK" "$ATLAS" "$COUNT_CSV" -stat_edge sum
  
  # --------- (4) LENGTH metrics (UNWEIGHTED) ---------
  local TOTAL_LEN="$OUTDIR/${SUBJ}_TOTAL_length.csv"
  echo "[Build] TOTAL_length (unweighted) -> $TOTAL_LEN"
  run_t2c_unweighted "$TCK" "$ATLAS" "$TOTAL_LEN" -scale_length -stat_edge sum
  
  local MEAN_LEN="$OUTDIR/${SUBJ}_MEAN_length.csv"
  echo "[Build] MEAN_length (unweighted) -> $MEAN_LEN"
  run_t2c_unweighted "$TCK" "$ATLAS" "$MEAN_LEN" -scale_length -stat_edge mean
  
  # --------- (5) Microstructure MEAN_{FA,AD,RD,MD} (weighted if TCKW present) ---------
  for TAG in FA MD; do
    local IMG_BASE="${SUBJ}_${TAG}"
    local IMG_PATH
    IMG_PATH="$(find_image "$IMG_BASE")" || true
    if [[ -n "$IMG_PATH" ]]; then
      make_metric_connectome "$SUBJ" "$TCK" "$ATLAS" "$IMG_PATH" "$TAG" "$OUTDIR"
    else
      echo "[Warn] ${IMG_BASE}.{mif|nii|nii.gz} not found — skipping MEAN_${TAG} and ${TAG}_perstream."
    fi
  done
  
  echo "[OK] $SUBJ done."
}

# --------- ENTRY POINT ---------
# Check prerequisites
[[ -f "$ATLAS_MNI" ]] || { echo "[ERROR] Atlas not found: $ATLAS_MNI"; exit 1; }
[[ -f "$MNI_T1" ]] || { echo "[ERROR] MNI T1 template not found: $MNI_T1"; exit 1; }
[[ -d "$PREPROC_ROOT" ]] || { echo "[ERROR] Preproc directory not found: $PREPROC_ROOT"; exit 1; }

ARG="$1"
if [[ "$ARG" == "all" ]]; then
  for s in "$PREPROC_ROOT"/sub-*; do
    [[ -d "$s" ]] && build_one "$(basename "$s")"
  done
else
  build_one "$ARG"
fi

echo "[Batch] Done."

