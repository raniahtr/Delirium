#!/usr/bin/env bash
# build_connectomes_426.sh
# Creates connectomes for a 426-parcel HCPex atlas per subject.
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
#   bash build_connectomes_426.sh sub-XX
#   bash build_connectomes_426.sh all
#
# Assumptions:
#   - Working tree: ROOT/sub-XX/ with:
#       Combined_HCPex_426.mif (float) OR Combined_HCPex_426_uint16.mif
#       ${SUBJ}_tracks_10M.tck
#       (optional) sift2_weights.txt or similar
#       (optional) ${SUBJ}_{FA,AD,RD,MD}.{mif|nii|nii.gz}
#   - MRtrix3 on PATH (tck2connectome, tcksample, mrstats, mrcalc, mrconvert)
#
set -euo pipefail

# --------- CONFIG ---------
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_new_clean"

# Atlas filenames (located in each subject directory)
ATLAS_UINT16="Combined_HCPex_426_uint16.mif"
ATLAS_FLOAT="Combined_HCPex_426.mif"

# Tractogram name pattern (update if yours differs)
TCK_NAME_SUFFIX="_tracks_10M.tck"

# tck->atlas assignment options
ASSIGN_OPTS=(-assignment_radial_search 3 -symmetric -zero_diagonal)

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

# Ensure atlas exists and is uint16; if only float exists, validate and cast
ensure_uint16_atlas(){
  # Globals used/set: ATLAS
  if [[ -f "$ATLAS_UINT16" ]]; then
    echo "[Info] Using $ATLAS_UINT16"
    ATLAS="$ATLAS_UINT16"
    return 0
  fi

  if [[ -f "$ATLAS_FLOAT" ]]; then
    echo "[Info] Found $ATLAS_FLOAT (float). Running QC before casting to uint16..."
    # Range check
    read -r MIN MAX <<<"$(mrstats "$ATLAS_FLOAT" -output min -output max)"
    echo "[QC] $ATLAS_FLOAT min=$MIN max=$MAX"
    # Allow small negative epsilon if interpolation left tiny negatives; clamp check via printf
    local MIN_INT
    MIN_INT=$(printf '%.0f\n' "$MIN")
    if [[ "$MAX" != "426" || "$MIN_INT" -lt 0 ]]; then
      echo "[ERROR] Atlas labels out of expected range [0..426]. Got min=$MIN max=$MAX"
      return 1
    fi
    # Integer-ness check (robust to whitespace & float noise)
    mrcalc "$ATLAS_FLOAT" -round "$ATLAS_FLOAT" -subtract _atlas_diff_tmp.mif -force

    # Trim whitespace and get the numeric max
    DIFFMAX_RAW="$(mrstats _atlas_diff_tmp.mif -output max)"
    DIFFMAX="$(echo "$DIFFMAX_RAW" | tr -d '[:space:]')"
    rm -f _atlas_diff_tmp.mif

    # Treat anything <= 1e-6 as zero (tolerance for tiny float noise)
    if awk -v x="$DIFFMAX" 'BEGIN{exit (x+0 > 1e-6 ? 0 : 1)}'; then
      echo "[ERROR] Atlas contains non-integer labels (max diff after rounding: $DIFFMAX); aborting cast."
      return 1
    fi

    # Safe cast
    mrconvert "$ATLAS_FLOAT" "$ATLAS_UINT16" -datatype uint16 -force
    echo "[OK] Casted to $ATLAS_UINT16"
    ATLAS="$ATLAS_UINT16"
    return 0
  fi

  echo "[ERROR] Neither $ATLAS_UINT16 nor $ATLAS_FLOAT found in $(pwd)"
  return 1
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

build_one(){
  local SUBJ="$1"
  local SDIR="$ROOT/$SUBJ"
  echo "=== $SUBJ ==="
  [[ -d "$SDIR" ]] || { echo "[ERROR] $SDIR not found"; return 1; }
  cd "$SDIR"

  # Ensure atlas present and in uint16 (or cast it safely)
  ensure_uint16_atlas || return 1

  # Optional final QC
  if command -v mrstats >/dev/null 2>&1; then
    read -r MIN MAX <<<"$(mrstats "$ATLAS" -output min -output max)"
    echo "[QC] $ATLAS min=$MIN max=$MAX"
    [[ "$MAX" == "426" ]] || { echo "[ERROR] Max label not 426"; return 1; }
  fi

  # Tractogram
  local TCK="${SUBJ}${TCK_NAME_SUFFIX}"
  [[ -f "$TCK" ]] || { echo "[ERROR] Missing $TCK"; return 1; }

  # SIFT2 weights (optional, but will upgrade weighted ops when present)
  TCKW=""
  for cand in \
      sift2_weights.txt \
      "${SUBJ}_sift2_weights.txt" \
      "${SUBJ}_sift_1M.txt" \
      "sub-${SUBJ#sub-}_sift_1M.txt" \
      "${SUBJ}_sift_weights.txt" ; do
    [[ -f "$cand" ]] && { TCKW="$cand"; break; }
  done

  if [[ -n "$TCKW" ]]; then
    echo "[Info] Using SIFT2 weights: $TCKW"
  else
    echo "[Warn] No SIFT2 weights found — SC_sift2 will fall back to counts; MEAN_* will be unweighted means."
  fi

  local OUTDIR="connectome_local"
  mkdir -p "$OUTDIR"

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
  for TAG in FA AD RD MD; do
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
ARG="$1"
if [[ "$ARG" == "all" ]]; then
  for s in "$ROOT"/sub-*; do
    [[ -d "$s" ]] && build_one "$(basename "$s")"
  done
else
  build_one "$ARG"
fi

echo "[Batch] Done."
