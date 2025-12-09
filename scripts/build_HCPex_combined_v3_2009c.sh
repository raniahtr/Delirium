#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Build Combined HCPex Atlas (Glasser 2009a → 2009c + HCPex 2009c)
# ============================================================================
# This script:
# 1. Transforms Glasser atlas from MNI152 ICBM2009a to 2009c using ANTs
# 2. Resamples HCPex to 1mm in 2009c space
# 3. Merges Glasser (1-360) + HCPex (361-426) in 2009c space
# 4. Transforms merged atlas to subject T1 then DWI space
# ============================================================================

# ---- Paths ----
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
PREPROC="$ROOT/Preproc_current"
ATLAS_DIR="$ROOT/atlas"
HCPEX_DIR="$ATLAS_DIR/HCPex_folder"

# Input atlases
GLASSER_2009A="$ATLAS_DIR/new_template/HCP-MMP1_on_MNI152_ICBM2009a_nlin_FIXED.nii.gz"
HCPEX_2MM=""  # Will be detected below

# Template files for ANTs registration
TEMPLATE_09A_GM="$ATLAS_DIR/mni_icbm152_nlin_asym_09a/mni_icbm152_gm_tal_nlin_asym_09a.nii"
TEMPLATE_09C_GM="$ATLAS_DIR/mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii"
TEMPLATE_09C_T1="$ATLAS_DIR/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii"

# Intermediate files
REG_DIR="$ATLAS_DIR/registration_09a_to_09c"
GLASSER_2009C="$ATLAS_DIR/HCP-MMP1_on_MNI152_ICBM2009c_nlin.nii.gz"
HCPEX_1MM_09C="$HCPEX_DIR/HCPex_1mm_2009c.nii.gz"

# Output: merged atlas in 2009c 1mm space
MERGED_ATLAS_09C="$ATLAS_DIR/Combined_HCPex_426_MNI2009c_1mm.nii.gz"

# Template for subject registration (use 2009c T1)
MNI_T1="$TEMPLATE_09C_T1"

# ---- Detect HCPex file name gracefully ----
HCPEX_CANDIDATES=("HCPex_2mm.nii" "HCPex_2mm.nii.gz" "HCPex.nii.gz" "HCPex.nii")
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

# ---- Check for ANTs ----
if ! command -v antsRegistrationSyN.sh &> /dev/null; then
  echo "[ERROR] ANTs not found. Please install ANTs or add it to PATH."
  echo "        Required: antsRegistrationSyN.sh, antsApplyTransforms"
  exit 1
fi

# ============================================================================
# STEP 1: Transform Glasser from 2009a to 2009c using ANTs
# ============================================================================
transform_glasser_09a_to_09c () {
  echo "=========================================="
  echo "STEP 1: Transforming Glasser (2009a → 2009c)"
  echo "=========================================="
  
  # Check inputs
  if [[ ! -f "$GLASSER_2009A" ]]; then
    echo "[ERROR] Glasser 2009a atlas not found: $GLASSER_2009A"
    exit 1
  fi
  
  if [[ ! -f "$TEMPLATE_09A_GM" ]]; then
    echo "[ERROR] 2009a GM template not found: $TEMPLATE_09A_GM"
    exit 1
  fi
  
  if [[ ! -f "$TEMPLATE_09C_GM" ]]; then
    echo "[ERROR] 2009c GM template not found: $TEMPLATE_09C_GM"
    exit 1
  fi
  
  # Check if transformation already exists
  if [[ -f "$GLASSER_2009C" ]]; then
    echo "[Reuse] Glasser 2009c already exists: $GLASSER_2009C"
    echo "        Delete it to rebuild."
    return
  fi
  
  # Create registration directory
  mkdir -p "$REG_DIR"
  cd "$REG_DIR"
  
  # Check if registration already done
  if [[ -f "reg_09a_to_09c_1Warp.nii.gz" && -f "reg_09a_to_09c_0GenericAffine.mat" ]]; then
    echo "[Reuse] Registration transforms already exist"
  else
    echo "[Register] Computing 2009a → 2009c transformation using ANTs..."
    echo "          This may take several minutes..."
    
    antsRegistrationSyN.sh \
      -d 3 \
      -f "$TEMPLATE_09C_GM" \
      -m "$TEMPLATE_09A_GM" \
      -o reg_09a_to_09c_ \
      -t s \
      -j 1 \
      -n 4
    
    if [[ ! -f "reg_09a_to_09c_1Warp.nii.gz" || ! -f "reg_09a_to_09c_0GenericAffine.mat" ]]; then
      echo "[ERROR] Registration failed. Check ANTs output above."
      exit 1
    fi
    echo "[OK] Registration complete"
  fi
  
  # Apply transformation to Glasser atlas
  echo "[Transform] Applying transformation to Glasser atlas..."
  antsApplyTransforms \
    -d 3 \
    -i "$GLASSER_2009A" \
    -r "$TEMPLATE_09C_GM" \
    -o "$GLASSER_2009C" \
    -n NearestNeighbor \
    -t reg_09a_to_09c_1Warp.nii.gz \
    -t reg_09a_to_09c_0GenericAffine.mat
  
  if [[ ! -f "$GLASSER_2009C" ]]; then
    echo "[ERROR] Transformation failed. Check ANTs output above."
    exit 1
  fi
  
  # QC: verify label ranges
  read gmin gmax <<<"$(fslstats "$GLASSER_2009C" -R)"
  echo "[QC] Glasser 2009c    (expect 1–360):   min=$gmin max=$gmax"
  
  if awk "BEGIN {exit !($gmin < 1 || $gmax > 360)}"; then
    echo "[WARNING] Glasser 2009c label range ($gmin-$gmax) is outside expected (1-360)"
  else
    echo "[OK] Glasser transformed successfully: $GLASSER_2009C"
  fi
  
  cd - >/dev/null
}

# ============================================================================
# STEP 2: Resample HCPex to 1mm in 2009c space
# ============================================================================
resample_hcpex_to_1mm_09c () {
  echo "=========================================="
  echo "STEP 2: Resampling HCPex to 1mm (2009c space)"
  echo "=========================================="
  
  if [[ ! -f "$GLASSER_2009C" ]]; then
    echo "[ERROR] Glasser 2009c not found. Run Step 1 first."
    exit 1
  fi
  
  if [[ -f "$HCPEX_1MM_09C" ]]; then
    echo "[Reuse] HCPex 1mm 2009c already exists: $HCPEX_1MM_09C"
    return
  fi
  
  # Create registration directory for HCPex alignment
  REG_HCPEX_DIR="$ATLAS_DIR/registration_hcpex_to_09c"
  mkdir -p "$REG_HCPEX_DIR"
  cd "$REG_HCPEX_DIR"
  
  # Registration matrix file
  HCPEX_TO_09C_MAT="$REG_HCPEX_DIR/hcpex_to_09c.mat"
  
  # Check if registration already done
  if [[ -f "$HCPEX_TO_09C_MAT" ]]; then
    echo "[Reuse] HCPex to 2009c registration already exists"
  else
    echo "[Register] Registering HCPex to 2009c T1 template for proper alignment..."
    echo "          This ensures HCPex is correctly aligned to 2009c space..."
    
    # Register HCPex to 2009c T1 template (affine, 12 DOF)
    # Use T1 template as reference for better alignment
    flirt -in "$HCPEX_2MM" \
          -ref "$TEMPLATE_09C_T1" \
          -omat "$HCPEX_TO_09C_MAT" \
          -dof 12 \
          -cost corratio \
          -searchrx -90 90 -searchry -90 90 -searchrz -90 90 \
          -interp nearestneighbour
    
    if [[ ! -f "$HCPEX_TO_09C_MAT" ]]; then
      echo "[ERROR] HCPex registration failed. Check FLIRT output above."
      cd - >/dev/null
      exit 1
    fi
    echo "[OK] Registration complete"
  fi
  
  # Apply transformation to resample HCPex to match Glasser 2009c grid exactly
  echo "[Resample] Resampling HCPex (2mm → 1mm) to match Glasser 2009c grid..."
  flirt -in "$HCPEX_2MM" \
        -ref "$GLASSER_2009C" \
        -applyxfm -init "$HCPEX_TO_09C_MAT" \
        -interp nearestneighbour \
        -out "$HCPEX_1MM_09C"
  
  if [[ ! -f "$HCPEX_1MM_09C" ]]; then
    echo "[ERROR] Resampling failed"
    cd - >/dev/null
    exit 1
  fi
  
  # QC: verify label ranges
  read hmin hmax <<<"$(fslstats "$HCPEX_1MM_09C" -R)"
  echo "[QC] HCPex 1mm 2009c  (expect 1–426):   min=$hmin max=$hmax"
  echo "[OK] HCPex resampled and aligned: $HCPEX_1MM_09C"
  
  cd - >/dev/null
}

# ============================================================================
# STEP 3: Merge Glasser (1-360) + HCPex (361-426) in 2009c space
# ============================================================================
build_merged_atlas_09c () {
  echo "=========================================="
  echo "STEP 3: Merging atlases in 2009c space"
  echo "=========================================="
  
  if [[ ! -f "$GLASSER_2009C" ]]; then
    echo "[ERROR] Glasser 2009c not found: $GLASSER_2009C"
    exit 1
  fi
  
  if [[ ! -f "$HCPEX_1MM_09C" ]]; then
    echo "[ERROR] HCPex 1mm 2009c not found: $HCPEX_1MM_09C"
    exit 1
  fi
  
  if [[ -f "$MERGED_ATLAS_09C" ]]; then
    echo "[Reuse] Merged atlas already exists: $MERGED_ATLAS_09C"
    echo "        Delete it to rebuild."
    return
  fi
  
  echo "[Merge] Combining Glasser (1-360) + HCPex (361-426) in 2009c space..."
  
  # Extract HCPex subcortical labels (>360)
  HCPEX_SUB_TEMP="$ATLAS_DIR/hcpex_sub_temp.nii.gz"
  fslmaths "$HCPEX_1MM_09C" -thr 361 -uthr 426 "$HCPEX_SUB_TEMP"
  
  # Merge: take maximum (Glasser 1-360 + HCPex 361-426, no overlap)
  MERGED_TEMP="$ATLAS_DIR/merged_atlas_temp.nii.gz"
  fslmaths "$GLASSER_2009C" -max "$HCPEX_SUB_TEMP" "$MERGED_TEMP"
  
  # Convert to uint16 for efficiency and consistency
  echo "[Convert] Converting merged atlas to uint16..."
  mrconvert "$MERGED_TEMP" "$MERGED_ATLAS_09C" -datatype uint16 -force -quiet
  
  # Cleanup temp files
  rm -f "$HCPEX_SUB_TEMP" "$MERGED_TEMP"
  
  # QC: verify label ranges
  read gmin gmax <<<"$(fslstats "$GLASSER_2009C" -R)"
  read hmin hmax <<<"$(fslstats "$HCPEX_1MM_09C" -R)"
  read cmin cmax <<<"$(fslstats "$MERGED_ATLAS_09C" -R)"
  echo "[QC] Glasser 2009c    (expect 1–360):   min=$gmin max=$gmax"
  echo "[QC] HCPex 1mm 2009c  (expect 1–426):   min=$hmin max=$hmax"
  echo "[QC] Merged 2009c     (expect 1–426):   min=$cmin max=$cmax"
  
  if awk "BEGIN {exit !($cmin < 1 || $cmax > 426)}"; then
    echo "[WARNING] Merged atlas label range ($cmin-$cmax) is outside expected (1-426)"
  else
    echo "[OK] Merged atlas created successfully: $MERGED_ATLAS_09C"
  fi
}

# ============================================================================
# STEP 4: Transform merged atlas to subject space (T1 then DWI)
# ============================================================================
process_subject () {
  local SUBJ="$1"
  local FORCE="${2:-}"  # Second argument: "force" or empty
  local SDIR="$PREPROC/$SUBJ"
  [[ -d "$SDIR" ]] || { echo "[Skip] $SUBJ: no dir"; return; }
  cd "$SDIR"

  echo -e "\n=== $SUBJ ==="

  # Check if merged atlas exists
  if [[ ! -f "$MERGED_ATLAS_09C" ]]; then
    echo "[ERROR] Merged atlas not found: $MERGED_ATLAS_09C"
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
  if [[ -f Combined_HCPex_426_2009c.mif ]] && [[ "$FORCE" != "force" ]]; then
    echo "[Reuse] Found Combined_HCPex_426_2009c.mif - delete to reprocess (or use 'force' flag)"
    cd - >/dev/null; return
  fi
  
  # If force flag is set and file exists, remove it first
  if [[ -f Combined_HCPex_426_2009c.mif ]] && [[ "$FORCE" == "force" ]]; then
    echo "[Force] Removing existing Combined_HCPex_426_2009c.mif to rebuild..."
    rm -f Combined_HCPex_426_2009c.mif Combined_HCPex_426_2009c.nii.gz
  fi

  # ---------- 2009c → T1 (affine) ----------
  echo "[Warp] Estimating 2009c→T1 affine"
  flirt -in "$MNI_T1" -ref T1.nii.gz -omat MNI2009c_to_T1.mat -dof 12 -cost corratio

  # ---------- Transform merged atlas: 2009c → T1 ----------
  echo "[Map] Merged atlas (2009c) → T1"
  flirt -in "$MERGED_ATLAS_09C" -ref T1.nii.gz -applyxfm -init MNI2009c_to_T1.mat \
        -interp nearestneighbour -out merged_atlas_in_T1.nii.gz

  # ---------- Transform merged atlas: T1 → DWI ----------
  echo "[Map] Merged atlas (T1) → DWI"
  mrconvert merged_atlas_in_T1.nii.gz merged_atlas_in_T1.mif -quiet -force
  transformcalc diff2struct_mrtrix.txt invert struct2diff.txt -force
  mrtransform merged_atlas_in_T1.mif -linear struct2diff.txt \
              -template mean_b0.mif -interp nearest merged_atlas_in_dwi.mif -quiet -force

  # Clean to uint16: round to nearest integer, preserving labels 1-426
  mrcalc merged_atlas_in_dwi.mif 0 -max merged_nonneg.mif -quiet -force
  mrcalc merged_nonneg.mif 0.5 -add -floor -datatype uint16 Combined_HCPex_426_2009c.mif -quiet -force
  rm -f merged_nonneg.mif

  # Convert to NIfTI for QC
  mrconvert Combined_HCPex_426_2009c.mif Combined_HCPex_426_2009c.nii.gz -quiet -force

  # ---------- QC ----------
  read cmin cmax <<<"$(fslstats Combined_HCPex_426_2009c.nii.gz -R)"
  echo "[QC] Combined in DWI    (expect 1–426): min=$cmin max=$cmax"
  
  if awk "BEGIN {exit !($cmin < 1 || $cmax > 426)}"; then
    echo "[WARNING] Label range ($cmin-$cmax) is outside expected (1-426)"
  fi

  # Optional connectome
  if [[ -f tracks_10M.tck ]]; then
    echo "[Connectome] tck2connectome → connectome_426_2009c.csv"
    tck2connectome tracks_10M.tck Combined_HCPex_426_2009c.mif connectome_426_2009c.csv \
      -assignment_radial_search 2 -force -quiet
  fi

  # Cleanup
  rm -f merged_atlas_in_T1.nii.gz merged_atlas_in_T1.mif struct2diff.txt MNI2009c_to_T1.mat
  cd - >/dev/null
}

# ============================================================================
# DRIVER: Handle different command-line arguments
# ============================================================================
FORCE_FLAG="${2:-}"  # Second argument: "force" or empty

case "${1:-}" in
  "build")
    # Steps 1-3: Transform Glasser, resample HCPex, merge
    transform_glasser_09a_to_09c
    echo ""
    resample_hcpex_to_1mm_09c
    echo ""
    build_merged_atlas_09c
    ;;
  "all")
    # Step 4 only: Process all subjects (assumes merged atlas exists)
    if [[ "$FORCE_FLAG" == "force" ]]; then
      echo "[Force mode] Will rebuild all existing connectomes"
    fi
    cd "$PREPROC"
    for s in sub-*; do [[ -d "$s" ]] && process_subject "$s" "$FORCE_FLAG"; done
    ;;
  "build+all")
    # All steps: Build atlas, then process all subjects
    transform_glasser_09a_to_09c
    echo ""
    resample_hcpex_to_1mm_09c
    echo ""
    build_merged_atlas_09c
    echo ""
    if [[ "$FORCE_FLAG" == "force" ]]; then
      echo "[Force mode] Will rebuild all existing connectomes"
    fi
    cd "$PREPROC"
    for s in sub-*; do [[ -d "$s" ]] && process_subject "$s" "$FORCE_FLAG"; done
    ;;
  sub-*)
    # Step 4 only: Process single subject (assumes merged atlas exists)
    process_subject "$1" "$FORCE_FLAG"
    ;;
  *)
    echo "Usage: bash $(basename "$0") <command> [force]"
    echo ""
    echo "Commands:"
    echo "  build          - Transform Glasser (2009a→2009c), resample HCPex, merge (one-time setup)"
    echo "  all            - Transform merged atlas to DWI space for all subjects"
    echo "  all force      - Transform merged atlas to DWI space for all subjects (rebuild existing)"
    echo "  build+all      - Build atlas, then process all subjects"
    echo "  build+all force - Build atlas, then process all subjects (rebuild existing)"
    echo "  sub-XX         - Transform merged atlas to DWI space for single subject"
    echo "  sub-XX force   - Transform merged atlas to DWI space for single subject (rebuild existing)"
    echo ""
    echo "This script uses:"
    echo "  - Glasser in 2009a space → transformed to 2009c using ANTs"
    echo "  - HCPex in 2009c space (native)"
    echo "  - Merged atlas in 2009c space"
    echo "  - 2009c T1 template for subject registration"
    echo ""
    echo "Example workflow:"
    echo "  1. bash $(basename "$0") build          # One-time setup (may take 10-30 min)"
    echo "  2. bash $(basename "$0") all            # Process all subjects"
    exit 1
    ;;
esac

echo "[Batch] Done."

