#!/usr/bin/env bash
# sizecorr.sh
# Calculates node volumes from the registered atlas and applies size correction to connectomes.
# Uses the atlas_in_dwi_uint16.mif created by build_final_connectomes.sh (432 labels).
#
# Usage:
#   bash sizecorr.sh [MATRIX_BASENAME]
#   bash sizecorr.sh SC_sift2
#   bash sizecorr.sh COUNT
#
# If MATRIX_BASENAME is not provided, defaults to SC_sift2
#
set -euo pipefail

ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_current"
MATRIX_BASE="${1:-SC_sift2}"  # Default to SC_sift2 if not provided
EXPECTED_MAX_LABEL=432

for s in "$ROOT"/sub-*; do
  SUBJ=$(basename "$s")
  [[ -d "$s" ]] || continue
  
  cd "$s"
  echo "=== $SUBJ ==="
  
  # Check if atlas exists (should be created by build_final_connectomes.sh)
  ATLAS="connectome_local/atlas_in_dwi_uint16.mif"
  if [[ ! -f "$ATLAS" ]]; then
    echo "[WARN] Atlas not found: $ATLAS"
    echo "       Run build_final_connectomes.sh first to create the registered atlas."
    cd - >/dev/null
    continue
  fi
  
  # Check if connectome exists
  CONNECTOME="connectome_local/${SUBJ}_${MATRIX_BASE}.csv"
  if [[ ! -f "$CONNECTOME" ]]; then
    echo "[WARN] Connectome not found: $CONNECTOME"
    echo "       Skipping $SUBJ"
    cd - >/dev/null
    continue
  fi
  
  # Calculate node volumes if not already done
  if [[ ! -f "connectome_local/${SUBJ}_node_volumes.csv" ]]; then
    echo "[Info] Calculating node volumes from registered atlas..."
    N=$(mrstats "$ATLAS" -output max | awk '{print int($1)}')
    
    # Verify label count
    if [[ "$N" != "$EXPECTED_MAX_LABEL" ]]; then
      echo "[WARN] Atlas max label ($N) does not match expected ($EXPECTED_MAX_LABEL)"
      echo "       Continuing anyway..."
    fi
    
    mkdir -p connectome_local
    echo "label,voxels" > "connectome_local/${SUBJ}_node_volumes.csv"
    
    for i in $(seq 1 $N); do
      # Create temporary mask for label i, count voxels
      TEMP_MASK="connectome_local/_temp_mask_${i}.mif"
      mrcalc "$ATLAS" $i -eq "$TEMP_MASK" -force -quiet 2>/dev/null || true
      vox=$(mrstats "$ATLAS" -mask "$TEMP_MASK" -output count 2>/dev/null | awk '{print $1}' || echo "0")
      rm -f "$TEMP_MASK" 2>/dev/null || true
      echo "$i,$vox" >> "connectome_local/${SUBJ}_node_volumes.csv"
    done
    echo "[OK] Node volumes calculated: connectome_local/${SUBJ}_node_volumes.csv"
  else
    echo "[Info] Node volumes already exist: connectome_local/${SUBJ}_node_volumes.csv"
  fi
  
  # Apply size correction
  echo "[Info] Applying size correction to ${MATRIX_BASE}..."
  python3 /media/RCPNAS/Data/Delirium/Delirium_Rania/scripts/sizecorr.py "${SUBJ}" "${MATRIX_BASE}"
  
  cd - >/dev/null
done

echo "[Batch] Done."
