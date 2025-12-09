#!/usr/bin/env bash
set -euo pipefail

# ================= CONFIG =================
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
PREPROC="${ROOT}/Preproc_new"
ASSIGN_RADIAL=3
MINWEIGHT=0.001
FORCE=1  # set 0 to avoid overwriting
# ==========================================

log(){ echo -e "\n[CONNECTOME] $*"; }

# Move into Preproc folder where all sub-* directories live
cd "$PREPROC" || { echo "Preproc folder not found: $PREPROC"; exit 1; }

for sdir in sub-*; do
  [[ -d "$sdir" ]] || continue
  cd "$sdir"

  subj="$sdir"
  tck="${subj}_tracks_10M.tck"
  w_in="${subj}_sift_1M.txt"
  tck_f="${subj}_filtered_v2.tck"
  w_out="${subj}_sift_filtered_v2.txt"
  atlas="atlas_in_dwi_uint16.mif"
  outdir="connectome_local"
  out_csv="${outdir}/${subj}_SC_sift2.csv"

  # --- Input checks ---
  missing=0
  [[ -f "$tck"   ]] || { echo "[Skip] Missing $tck";   missing=1; }
  [[ -f "$w_in"  ]] || { echo "[Skip] Missing $w_in";  missing=1; }
  [[ -f "$atlas" ]] || { echo "[Skip] Missing $atlas"; missing=1; }
  if (( missing )); then cd ..; continue; fi

  mkdir -p "$outdir"
  log "Processing $subj"

  # --- Export filtered tractogram + matching weights ---
  if [[ ! -f "$tck_f" || ! -f "$w_out" || "$FORCE" -eq 1 ]]; then
    tckedit "$tck" "$tck_f" \
      -tck_weights_in  "$w_in" \
      -tck_weights_out "$w_out" \
      -minweight "$MINWEIGHT" ${FORCE:+-force}
  else
    echo "[Info] $tck_f and $w_out exist → keeping them (FORCE=0)."
  fi

  # --- Build connectome ---
  if [[ ! -f "$out_csv" || "$FORCE" -eq 1 ]]; then
    tck2connectome "$tck_f" "$atlas" "$out_csv" \
      -tck_weights_in "$w_out" \
      -symmetric -zero_diagonal \
      -assignment_radial_search "$ASSIGN_RADIAL" ${FORCE:+-force}
  else
    echo "[Info] $out_csv already exists → keeping it (FORCE=0)."
  fi

  # --- Tiny QC: density summary ---
  python3 - "$out_csv" <<'PY'
import sys, numpy as np
p = sys.argv[1]
M = np.loadtxt(p, delimiter=",")
nnz = (M!=0).sum()
empties = np.where((M.sum(0)+M.sum(1))==0)[0]
print(f"[QC] {p}: nnz={nnz}, density={nnz/M.size:.3%}, empty_nodes={len(empties)}")
PY

  cd ..
done

echo -e "\n[CONNECTOME] All subjects processed."

