#!/usr/bin/env bash
set -euo pipefail

# ---- Paths ----
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
PREPROC="$ROOT/Preproc_new_clean"
ATLAS_DIR="$ROOT/atlas"
ATLAS_GLR="$ATLAS_DIR/HCPMMP1_labels_3D.nii.gz"          # Glasser (1mm)
HCPEX_DIR="$ATLAS_DIR/HCPex_folder"

# Accept any of these; we'll normalize below
HCPEX_CANDIDATES=("HCPex_2mm.nii" "HCPex_2mm.nii.gz" "HCPex.nii.gz" "HCPex.nii")
HCPEX_2MM=""     # will be set after detection
HCPEX_1MM="$HCPEX_DIR/HCPex_1mm_resampled.nii.gz"

# Output: merged atlas in MNI 1mm space
MERGED_ATLAS_MNI="$ATLAS_DIR/Combined_HCPex_426_MNI_1mm.nii.gz"

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

# ============================================================================
# STEP 1: Build merged atlas in MNI 1mm space (one-time setup)
# ============================================================================
build_merged_atlas () {
  echo "=========================================="
  echo "STEP 1: Building merged atlas in MNI 1mm space"
  echo "=========================================="
  
  # Check if Glasser exists
  if [[ ! -f "$ATLAS_GLR" ]]; then
    echo "[ERROR] Glasser atlas not found: $ATLAS_GLR"
    exit 1
  fi
  
  # Ensure HCPex 1mm exists (resample once)
  if [[ ! -f "$HCPEX_1MM" ]]; then
    echo "[Prep] Resampling HCPex (→ 1mm) to match Glasser..."
    flirt -in "$HCPEX_2MM" \
          -ref "$ATLAS_GLR" \
          -applyisoxfm 1 \
          -interp nearestneighbour \
          -out "$HCPEX_1MM"
    echo "[Prep] Created: $HCPEX_1MM"
  else
    echo "[Reuse] HCPex 1mm already exists: $HCPEX_1MM"
  fi
  
  # Check if merged atlas already exists
  if [[ -f "$MERGED_ATLAS_MNI" ]]; then
    echo "[Reuse] Merged atlas already exists: $MERGED_ATLAS_MNI"
    echo "        Delete it to rebuild."
    return
  fi
  
  echo "[Merge] Combining Glasser (1-360) + HCPex (>360) in MNI 1mm space..."
  
  # Convert to NIfTI for easier manipulation with FSL
  # Glasser is already 1-360, keep as is
  # HCPex: keep only labels > 360 (subcortical regions)
  HCPEX_SUB_TEMP="$ATLAS_DIR/hcpex_sub_temp.nii.gz"
  fslmaths "$HCPEX_1MM" -thr 361 -uthr 426 "$HCPEX_SUB_TEMP"
  
  # Merge: take maximum (Glasser 1-360 + HCPex 361-426, no overlap)
  MERGED_TEMP="$ATLAS_DIR/merged_atlas_temp.nii.gz"
  fslmaths "$ATLAS_GLR" -max "$HCPEX_SUB_TEMP" "$MERGED_TEMP"
  
  # Convert to uint16 for efficiency and consistency (labels are integers 0-426)
  echo "[Convert] Converting merged atlas to uint16..."
  mrconvert "$MERGED_TEMP" "$MERGED_ATLAS_MNI" -datatype uint16 -force -quiet
  
  # Cleanup temp files
  rm -f "$HCPEX_SUB_TEMP" "$MERGED_TEMP"
  
  # QC: verify label ranges
  read gmin gmax <<<"$(fslstats "$ATLAS_GLR" -R)"
  read hmin hmax <<<"$(fslstats "$HCPEX_1MM" -R)"
  read cmin cmax <<<"$(fslstats "$MERGED_ATLAS_MNI" -R)"
  echo "[QC] Glasser (MNI)      (expect 1–360):   min=$gmin max=$gmax"
  echo "[QC] HCPex (MNI)        (expect 1–426):   min=$hmin max=$hmax"
  echo "[QC] Merged (MNI)       (expect 1–426):   min=$cmin max=$cmax"
  
  # QC: Check label range (using awk for floating point comparison)
  if awk "BEGIN {exit !($cmin < 1 || $cmax > 426)}"; then
    echo "[WARNING] Merged atlas label range ($cmin-$cmax) is outside expected (1-426)"
  else
    echo "[OK] Merged atlas created successfully: $MERGED_ATLAS_MNI"
  fi
}

# ============================================================================
# STEP 2: Transform merged atlas to subject space (T1 then DWI)
# ============================================================================
process_subject () {
  local SUBJ="$1"
  local FORCE="${2:-}"  # Second argument: "force" or empty
  local SDIR="$PREPROC/$SUBJ"
  [[ -d "$SDIR" ]] || { echo "[Skip] $SUBJ: no dir"; return; }
  cd "$SDIR"

  echo -e "\n=== $SUBJ ==="

  # Check if merged atlas exists
  if [[ ! -f "$MERGED_ATLAS_MNI" ]]; then
    echo "[ERROR] Merged atlas not found: $MERGED_ATLAS_MNI"
    echo "        Run with 'build' argument first to create it."
    cd - >/dev/null
    return
  fi

  # Required inputs
  if [[ ! -f mean_b0.mif || ! -f diff2struct_mrtrix.txt ]]; then
    echo "[Skip] Missing mean_b0.mif or diff2struct_mrtrix.txt"
    cd - >/dev/null; return
  fi
  [[ -f T1.nii.gz ]] || { [[ -f T1.mif ]] && mrconvert T1.mif T1.nii.gz -force -quiet; }
  [[ -f T1.nii.gz ]] || { echo "[Skip] $SUBJ: no T1"; cd - >/dev/null; return; }

  # Check if output already exists (skip if force flag not set)
  if [[ -f Combined_HCPex_426.mif ]] && [[ "$FORCE" != "force" ]]; then
    echo "[Reuse] Found Combined_HCPex_426.mif - delete to reprocess (or use 'force' flag)"
    cd - >/dev/null; return
  fi
  
  # If force flag is set and file exists, remove it first
  if [[ -f Combined_HCPex_426.mif ]] && [[ "$FORCE" == "force" ]]; then
    echo "[Force] Removing existing Combined_HCPex_426.mif to rebuild..."
    rm -f Combined_HCPex_426.mif Combined_HCPex_426.nii.gz
  fi

  # ---------- MNI -> T1 (affine) ----------
  echo "[Warp] Estimating MNI→T1 affine"
  flirt -in "$MNI_T1" -ref T1.nii.gz -omat MNItoT1.mat -dof 12 -cost corratio

  # ---------- Transform merged atlas: MNI → T1 ----------
  echo "[Map] Merged atlas (MNI) → T1"
  flirt -in "$MERGED_ATLAS_MNI" -ref T1.nii.gz -applyxfm -init MNItoT1.mat \
        -interp nearestneighbour -out merged_atlas_in_T1.nii.gz

  # ---------- Transform merged atlas: T1 → DWI ----------
  echo "[Map] Merged atlas (T1) → DWI"
  mrconvert merged_atlas_in_T1.nii.gz merged_atlas_in_T1.mif -quiet -force
  transformcalc diff2struct_mrtrix.txt invert struct2diff.txt -force
  mrtransform merged_atlas_in_T1.mif -linear struct2diff.txt \
              -template mean_b0.mif -interp nearest merged_atlas_in_dwi.mif -quiet -force

  # Clean to uint16: round to nearest integer, preserving labels 1-426
  mrcalc merged_atlas_in_dwi.mif 0 -max merged_nonneg.mif -quiet -force
  mrcalc merged_nonneg.mif 0.5 -add -floor -datatype uint16 Combined_HCPex_426.mif -quiet -force
  rm -f merged_nonneg.mif

  # Convert to NIfTI for QC
  mrconvert Combined_HCPex_426.mif Combined_HCPex_426.nii.gz -quiet -force

  # ---------- QC ----------
  read cmin cmax <<<"$(fslstats Combined_HCPex_426.nii.gz -R)"
  echo "[QC] Combined in DWI    (expect 1–426): min=$cmin max=$cmax"
  
  # QC: Check label range
  if awk "BEGIN {exit !($cmin < 1 || $cmax > 426)}"; then
    echo "[WARNING] Label range ($cmin-$cmax) is outside expected (1-426)"
  fi

  # Optional connectome
  if [[ -f tracks_10M.tck ]]; then
    echo "[Connectome] tck2connectome → connectome_426.csv"
    tck2connectome tracks_10M.tck Combined_HCPex_426.mif connectome_426.csv \
      -assignment_radial_search 2 -force -quiet
  fi

  # Cleanup
  rm -f merged_atlas_in_T1.nii.gz merged_atlas_in_T1.mif struct2diff.txt MNItoT1.mat
  cd - >/dev/null
}

# ============================================================================
# DRIVER: Handle different command-line arguments
# ============================================================================
FORCE_FLAG="${2:-}"  # Second argument: "force" or empty

case "${1:-}" in
  "build")
    # Step 1 only: Build merged atlas in MNI space
    build_merged_atlas
    ;;
  "all")
    # Step 2 only: Process all subjects (assumes merged atlas exists)
    if [[ "$FORCE_FLAG" == "force" ]]; then
      echo "[Force mode] Will rebuild all existing connectomes"
    fi
    cd "$PREPROC"
    for s in sub-*; do [[ -d "$s" ]] && process_subject "$s" "$FORCE_FLAG"; done
    ;;
  "build+all")
    # Both steps: Build atlas, then process all subjects
    build_merged_atlas
    echo ""
    if [[ "$FORCE_FLAG" == "force" ]]; then
      echo "[Force mode] Will rebuild all existing connectomes"
    fi
    cd "$PREPROC"
    for s in sub-*; do [[ -d "$s" ]] && process_subject "$s" "$FORCE_FLAG"; done
    ;;
  sub-*)
    # Step 2 only: Process single subject (assumes merged atlas exists)
    process_subject "$1" "$FORCE_FLAG"
    ;;
  *)
    echo "Usage: bash $(basename "$0") <command> [force]"
    echo ""
    echo "Commands:"
    echo "  build          - Build merged atlas in MNI 1mm space (one-time setup)"
    echo "  all            - Transform merged atlas to DWI space for all subjects"
    echo "  all force      - Transform merged atlas to DWI space for all subjects (rebuild existing)"
    echo "  build+all      - Build atlas, then process all subjects"
    echo "  build+all force - Build atlas, then process all subjects (rebuild existing)"
    echo "  sub-XX         - Transform merged atlas to DWI space for single subject"
    echo "  sub-XX force   - Transform merged atlas to DWI space for single subject (rebuild existing)"
    echo ""
    echo "Example workflow:"
    echo "  1. bash $(basename "$0") build          # One-time setup"
    echo "  2. bash $(basename "$0") all            # Process all subjects"
    echo "  3. bash $(basename "$0") all force      # Rebuild all existing connectomes"
    exit 1
    ;;
esac

echo "[Batch] Done."

