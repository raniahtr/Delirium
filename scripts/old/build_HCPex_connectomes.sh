#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_new_clean"
ATLAS_COMBINED="/media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Combined_HCPex_426_uint16.mif"

usage(){ echo "Usage: $(basename "$0") sub-XX | all"; exit 1; }
[[ $# -eq 1 ]] || usage

build_one() {
  local SUBJ="$1"
  local SDIR="$ROOT/$SUBJ"
  echo "=== $SUBJ ==="
  [[ -d "$SDIR" ]] || { echo "[ERROR] $SDIR not found"; return 1; }
  cd "$SDIR"

  # ---------- STRICT: require 426 atlas ----------

  # Quick sanity on label range (allow 0 background)
  if command -v mrstats >/dev/null 2>&1; then
    read -r MIN MAX <<<"$(mrstats "$ATLAS_COMBINED" -output min -output max)"
    echo "[QC] $ATLAS_COMBINED min=$MIN max=$MAX"
    [[ "$MAX" == "426" ]] || { echo "[ERROR] Max label not 426"; return 1; }
  fi

  # ---------- inputs ----------
  local TCK="${SUBJ}_tracks_10M.tck"
  [[ -f "$TCK" ]] || { echo "[ERROR] $TCK missing"; return 1; }

  # SIFT2 weights (optional but preferred) — try common names
  local TCKW=""
  for cand in sift2_weights.txt "${SUBJ}_sift_1M.txt" "sub-${SUBJ#sub-}_sift_1M.txt" sub-AF_sift_1M.txt; do
    [[ -f "$cand" ]] && { TCKW="$cand"; break; }
  done
  [[ -n "$TCKW" ]] && echo "[Info] Using SIFT2 weights: $TCKW" || echo "[Warn] No SIFT2 weights; using raw counts for _COUNT and edge means for metrics."

  mkdir -p connectome_local

  # helper to call tck2connectome consistently
  run_t2c() {
    local OUTCSV="$1"; shift
    if [[ -n "$TCKW" ]]; then
      tck2connectome "$TCK" "$ATLAS_COMBINED" "$OUTCSV" \
        -tck_weights_in "$TCKW" -assignment_radial_search 3 \
        -symmetric -zero_diagonal -force "$@"
    else
      tck2connectome "$TCK" "$ATLAS_COMBINED" "$OUTCSV" \
        -assignment_radial_search 3 -symmetric -zero_diagonal -force "$@"
    fi
  }

  # ---------- (1) Weighted count matrix ----------
#  run_t2c "connectome_local/${SUBJ}_COUNT_SIFT2.csv" -stat_edge sum
   WEIGHTS="${SUBJ}_sift_coeffs.txt"     # preferred
   [[ ! -f "$WEIGHTS" && -f "${SUBJ}_sift_1M.txt" ]] && WEIGHTS="${SUBJ}_sift_1M.txt"

   run_t2c "$TCK" "$ATLAS_COMBINED" "connectome_local/${SUBJ}_SC_SIFT2.csv" \
     -tck_weights_in "$WEIGHTS"

   # Quick QC
   rows=$(wc -l < "connectome_local/${SUBJ}_SC_SIFT2.csv")
   cols=$(awk -F',' 'NR==1{print NF}' "connectome_local/${SUBJ}_SC_SIFT2.csv")
   echo "[QC] ${SUBJ}_SC_SIFT2.csv -> ${rows}x${cols} (expect 426x426)"
  # Also raw streamline counts (descriptive)
#  tck2connectome "$TCK" "$ATLAS_COMBINED" "connectome_local/${SUBJ}_COUNT.csv" \
#    -assignment_radial_search 3 -symmetric -zero_diagonal -force

  # ---------- (2) Microstructure edge means via per-streamline sampling ----------
#  make_metric_connectome () {
#    local IMG="$1" TAG="$2"
#    local SCALES="connectome_local/${SUBJ}_${TAG}_perstream.txt"
#    tcksample "$TCK" "$IMG" "$SCALES" -stat_tck mean -force -quiet
#    run_t2c "connectome_local/${SUBJ}_MEAN_${TAG}.csv" -scale_file "$SCALES" -stat_edge mean
#  }

#  for TAG in FA MD AD RD; do
#    [[ -f "${SUBJ}_${TAG}.nii" ]] && make_metric_connectome "${SUBJ}_${TAG}.nii" "$TAG"
#    [[ -f "${SUBJ}_${TAG}.mif" ]] && make_metric_connectome "${SUBJ}_${TAG}.mif" "$TAG"
#  done

  # ---------- lightweight QC ----------
  local M="connectome_local/${SUBJ}_COUNT_SIFT2.csv"
  if [[ -f "$M" ]]; then
    local NCOLS; NCOLS=$(head -n1 "$M" | awk -F, '{print NF}')
    echo "[QC] columns=$NCOLS (expect 426)"
    [[ "$NCOLS" -eq 426 ]] || echo "[WARN] matrix is not 426×426"
  fi

  cd - >/dev/null
}

ARG="$1"
if [[ "$ARG" == "all" ]]; then
  for s in "$ROOT"/sub-*; do [[ -d "$s" ]] && build_one "$(basename "$s")"; done
else
  build_one "$ARG"
fi

echo "[Batch] Done."
