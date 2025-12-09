#!/usr/bin/env bash
set -euo pipefail

# =====================  TUNABLES =====================
TOPK=${TOPK:-30}          # How many strongest ROI pairs to visualize
PER_EDGE=${PER_EDGE:-150} # Max streamlines per ROI pair
FORCE=${FORCE:-1}         # 1: overwrite existing viz exports, 0: skip if exist

# Helper: echo with tag
log(){ echo -e "\n[EXPORT] $*"; }

# Figure out where we are; require running from Preproc
PWD_BN="$(basename "$PWD")"
if [[ "$PWD_BN" != "Preproc_new" ]]; then
  echo "Please run this script from: .../Delirium_Rania/Preproc"
  echo "   (Current: $PWD)"
  exit 1
fi

# If a subject is passed (e.g., sub-DA), process only that one; else loop all sub-*
SUBJS=("$@")
if [[ ${#SUBJS[@]} -eq 0 ]]; then
  mapfile -t SUBJS < <(ls -d sub-* 2>/dev/null | xargs -n1 basename)
fi

if [[ ${#SUBJS[@]} -eq 0 ]]; then
  echo "No subject directories found (sub-*)"
  exit 1
fi

for SUBJ in "${SUBJS[@]}"; do
  [[ -d "$SUBJ" ]] || { echo "[Skip] $SUBJ is not a directory"; continue; }
  cd "$SUBJ"

  log "Subject: $SUBJ (TOPK=$TOPK, PER_EDGE=$PER_EDGE)"

  # Inputs (already produced by the pipeline) 
  TCK="${SUBJ}_filtered_v2.tck"
  WTS="${SUBJ}_sift_filtered_v2.txt"         # matches the filtered_v2.tck
  CSV="connectome_local/${SUBJ}_SC_sift2.csv" # connectome
  ATLAS="atlas_in_dwi_uint16.mif"
  B0="mean_b0.mif"
  ASS="connectome_local/${SUBJ}_assignments.csv"   # will create if missing

  #  Checks 
  missing=0
  for req in "$TCK" "$WTS" "$CSV" "$ATLAS" "$B0"; do
    [[ -f "$req" ]] || { echo "   [Missing] $req"; missing=1; }
  done
  if (( missing )); then
    echo "   ➜ Skipping $SUBJ (missing inputs)"
    cd ..
    continue
  fi

  # Outputs
  OUTDIR="viz_exports"
  SEL="$OUTDIR/${SUBJ}_select_weights.txt"
  TCK_OUT="$OUTDIR/${SUBJ}_viz_subset.tck"
  ATLAS_NII="$OUTDIR/${SUBJ}_atlas_in_dwi_uint16.nii.gz"
  B0_NII="$OUTDIR/${SUBJ}_mean_b0.nii.gz"
  
  mkdir -p "$OUTDIR"

  # Generate assignments if needed (streamline -> ROI pair)
  if [[ ! -f "$ASS" || "$FORCE" -eq 1 ]]; then
    TMPMAT="$OUTDIR/${SUBJ}_ignore.csv" #temporary matrix
    tck2connectome "$TCK" "$ATLAS" "$TMPMAT" \
      -tck_weights_in "$WTS" \
      -assignment_radial_search 3 \
      -out_assignments "$ASS" -force
    rm -f "$TMPMAT"

  else
    echo "   [Info] Using existing assignments: $ASS"
  fi

  # Python: compute selection mask (0/1 per streamline)
  if [[ ! -f "$SEL" || "$FORCE" -eq 1 ]]; then
    python3 - <<PY
import numpy as np, csv
import sys
CSV_PATH = "$CSV"
ASS_PATH = "$ASS"
TOPK     = int("$TOPK")
PER_EDGE = int("$PER_EDGE")

W = np.loadtxt(CSV_PATH, delimiter=',')
np.fill_diagonal(W, 0.0)
iu = np.triu_indices_from(W, 1)
pairs = [(i+1,j+1,float(W[i,j])) for i,j in zip(*iu) if W[i,j]>0]
pairs.sort(key=lambda x: x[2], reverse=True)
top = set((i,j) for (i,j,_) in pairs[:TOPK])

# Read assignments; accept 2 or 3 columns per line
sel_pairs = []
with open(ASS_PATH, 'r') as f:
    r = csv.reader(f)
    for row in r:
        if not row or row[0].startswith('#'):
            continue
        vals = [int(float(x)) for x in row]
        if len(vals)==3:
            _, a, b = vals
        else:
            a, b = vals
        i, j = sorted((a,b))
        sel_pairs.append((i,j))

from collections import defaultdict
counts = defaultdict(int)
keep = np.zeros(len(sel_pairs), dtype=np.float32)
for idx, (i,j) in enumerate(sel_pairs):
    if (i,j) in top and counts[(i,j)] < PER_EDGE:
        keep[idx] = 1.0
        counts[(i,j)] += 1

print(f"   [Py] Selected {int(keep.sum())} streamlines across {len(top)} edges.")
np.savetxt("$SEL", keep, fmt="%.0f")
PY
  else
    echo "   [Info] Using existing selection: $SEL"
  fi

  # Create tiny viz tractogram using selection as mask
  if [[ ! -f "$TCK_OUT" || "$FORCE" -eq 1 ]]; then
    tckedit "$TCK" "$TCK_OUT" \
      -tck_weights_in "$SEL" \
      -minweight 0.5 -force
  else
    echo "   [Info] Using existing viz subset: $TCK_OUT"
  fi

  # Convert small supporting volumes to NIfTI for easy local viz
  if [[ ! -f "$ATLAS_NII" || "$FORCE" -eq 1 ]]; then
    mrconvert "$ATLAS" "$ATLAS_NII" -force
  fi
  if [[ ! -f "$B0_NII" || "$FORCE" -eq 1 ]]; then
    mrconvert "$B0" "$B0_NII" -force
  fi

  # Summary 
  echo "   [Done] Created:"
  ls -lh "$TCK_OUT" "$ATLAS_NII" "$B0_NII" "$SEL" | sed 's/^/   • /'

  cd ..
done

echo -e "\n[EXPORT] Finished."
