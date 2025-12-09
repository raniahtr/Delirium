#!/bin/bash

###############################################################################
# DWI Preprocessing Pipeline (Batch)
#
# ⚠️ IMPORTANT:
# This script assumes the data has been organized and renamed by:
#     ./Organize.sh
#
# Directory structure should be:
#   Preproc/
#     ├── sub-001/
#     │     ├── sub-001_AP_A_dwi.nii.gz
#     │     ├── sub-001_AP_P_dwi.nii.gz (optional)
#     │     ├── sub-001_T1w.nii.gz
#     │     └── ...
###############################################################################
# -------- Setup ANTs path if available --------------------------------------
# Check for ANTs in common locations and add to PATH if found
if ! command -v N4BiasFieldCorrection >/dev/null 2>&1; then
  # Common ANTs installation locations
  ANTS_PATHS=(
    "/opt/ANTs/bin"
    "/usr/local/ants/bin"
    "/usr/local/ANTs/bin"
    "$HOME/ants/bin"
    "$HOME/ANTs/bin"
  )
  
  for ants_path in "${ANTS_PATHS[@]}"; do
    if [[ -f "$ants_path/N4BiasFieldCorrection" ]]; then
      export PATH="$ants_path:$PATH"
      echo "Found ANTs at: $ants_path (added to PATH)"
      break
    fi
  done
fi
# -----------------------------------------------------------------------------

# -------- PATCH A (helper) ---------------------------------------------------
# Read TRT (TotalReadoutTime) from JSON; else compute from EES & ReconMatrixPE; else fallback
read_trt_from_json() {
  local json="$1"
  if command -v jq >/dev/null 2>&1 && [[ -f "$json" ]]; then
    local trt
    trt=$(jq -r '.TotalReadoutTime // empty' "$json" 2>/dev/null)
    if [[ -n "$trt" && "$trt" != "null" ]]; then
      echo "$trt"; return 0
    fi
    local ees rpe
    ees=$(jq -r '.EffectiveEchoSpacing // .EstimatedEffectiveEchoSpacing // empty' "$json" 2>/dev/null)
    rpe=$(jq -r '.ReconMatrixPE // .AcquisitionMatrixPE // .PhaseEncodingSteps // empty' "$json" 2>/dev/null)
    if [[ -n "$ees" && -n "$rpe" && "$ees" != "null" && "$rpe" != "null" ]]; then
      awk -v ees="$ees" -v rpe="$rpe" 'BEGIN{printf "%.7f", ees*(rpe-1)}'
      return 0
    fi
  fi
  # Default fallback if nothing found
  echo "0.0450000"
  return 0
}

# Read Phase Encoding Direction from JSON and convert to eddy acqparams format
# Returns: "x y z" vector for acqparams.txt (e.g., "0 -1 0" for AP/j, "0 1 0" for PA/j-)
get_pe_vector_from_json() {
  local json="$1"
  local default_pe="$2"  # "AP" or "PA" as fallback
  local pe_dir=""
  
  if command -v jq >/dev/null 2>&1 && [[ -f "$json" ]]; then
    pe_dir=$(jq -r '.PhaseEncodingDirection // empty' "$json" 2>/dev/null)
  fi
  
  # Determine PE vector based on PhaseEncodingDirection
  # BIDS: "j" = AP (anterior-posterior, negative j), "j-" = PA (posterior-anterior, positive j)
  # eddy acqparams: "0 -1 0" = negative j (AP), "0 1 0" = positive j (PA)
  if [[ "$pe_dir" == "j" ]]; then
    echo "0 -1 0"
  elif [[ "$pe_dir" == "j-" ]]; then
    echo "0 1 0"
  elif [[ "$pe_dir" == "i" ]]; then
    echo "-1 0 0"
  elif [[ "$pe_dir" == "i-" ]]; then
    echo "1 0 0"
  elif [[ "$pe_dir" == "k" ]]; then
    echo "0 0 -1"
  elif [[ "$pe_dir" == "k-" ]]; then
    echo "0 0 1"
  else
    # Fallback: infer from default_pe or filename
    if [[ "$default_pe" == "AP" ]]; then
      echo "0 -1 0"
    elif [[ "$default_pe" == "PA" ]]; then
      echo "0 1 0"
    else
      # Default to AP if unknown
      echo "0 -1 0"
    fi
  fi
}
# ---------------------------------------------------------------------------

BASE_DIR="/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_current"
cd "$BASE_DIR" || { echo "ERROR: Could not cd to $BASE_DIR"; exit 1; }

# --------------------------------------------------------------------------
# Parse command-line arguments (optional subject specification)
# --------------------------------------------------------------------------
TARGET_SUBJECT=""
if [ $# -gt 0 ]; then
  TARGET_SUBJECT="$1"
  # Remove 'sub-' prefix if user provided it
  TARGET_SUBJECT="${TARGET_SUBJECT#sub-}"
  TARGET_SUBJECT="sub-${TARGET_SUBJECT}"
  
  # Verify subject exists
  if [ ! -d "$BASE_DIR/$TARGET_SUBJECT" ]; then
    echo "ERROR: Subject directory not found: $BASE_DIR/$TARGET_SUBJECT"
    echo "Available subjects:"
    ls -d sub-* 2>/dev/null | head -10
    exit 1
  fi
  echo "Processing single subject: $TARGET_SUBJECT"
else
  echo "Processing all subjects in $BASE_DIR"
fi

# --------------------------------------------------------------------------
# Compatibility layer: map PE-tagged files (AP_A, AP_P) to AP / PA names
# so the rest of the pipeline finds them.
# --------------------------------------------------------------------------
if [ -n "$TARGET_SUBJECT" ]; then
  SUBJECT_LIST=("$TARGET_SUBJECT")
else
  SUBJECT_LIST=(sub-*)
fi

for subj_folder in "${SUBJECT_LIST[@]}"; do
  [ -d "$subj_folder" ] || continue
  cd "$BASE_DIR/$subj_folder" || continue

  # Handle AP_A files (both .nii.gz and .nii)
  # Only create symlinks for files that actually exist
  if [ -f "${subj_folder}_AP_A_dwi.nii.gz" ]; then
    ln -sf "${subj_folder}_AP_A_dwi.nii.gz"  "${subj_folder}_AP_dwi.nii.gz"
    [ -f "${subj_folder}_AP_A_dwi.bval" ] && \
      ln -sf "${subj_folder}_AP_A_dwi.bval" "${subj_folder}_AP_dwi.bval"
    [ -f "${subj_folder}_AP_A_dwi.bvec" ] && \
      ln -sf "${subj_folder}_AP_A_dwi.bvec" "${subj_folder}_AP_dwi.bvec"
    [ -f "${subj_folder}_AP_A_dwi.json" ] && \
      ln -sf "${subj_folder}_AP_A_dwi.json" "${subj_folder}_AP_dwi.json"
  elif [ -f "${subj_folder}_AP_A_dwi.nii" ]; then
    ln -sf "${subj_folder}_AP_A_dwi.nii"  "${subj_folder}_AP_dwi.nii"
    [ -f "${subj_folder}_AP_A_dwi.bval" ] && \
      ln -sf "${subj_folder}_AP_A_dwi.bval" "${subj_folder}_AP_dwi.bval"
    [ -f "${subj_folder}_AP_A_dwi.bvec" ] && \
      ln -sf "${subj_folder}_AP_A_dwi.bvec" "${subj_folder}_AP_dwi.bvec"
    [ -f "${subj_folder}_AP_A_dwi.json" ] && \
      ln -sf "${subj_folder}_AP_A_dwi.json" "${subj_folder}_AP_dwi.json"
  fi

  # Handle AP_P files (both .nii.gz and .nii)
  # Only create symlinks for files that actually exist
  if [ -f "${subj_folder}_AP_P_dwi.nii.gz" ]; then
    ln -sf "${subj_folder}_AP_P_dwi.nii.gz"  "${subj_folder}_PA_dwi.nii.gz"
    [ -f "${subj_folder}_AP_P_dwi.bval" ] && \
      ln -sf "${subj_folder}_AP_P_dwi.bval" "${subj_folder}_PA_dwi.bval"
    [ -f "${subj_folder}_AP_P_dwi.bvec" ] && \
      ln -sf "${subj_folder}_AP_P_dwi.bvec" "${subj_folder}_PA_dwi.bvec"
    [ -f "${subj_folder}_AP_P_dwi.json" ] && \
      ln -sf "${subj_folder}_AP_P_dwi.json" "${subj_folder}_PA_dwi.json"
  elif [ -f "${subj_folder}_AP_P_dwi.nii" ]; then
    ln -sf "${subj_folder}_AP_P_dwi.nii"  "${subj_folder}_PA_dwi.nii"
    [ -f "${subj_folder}_AP_P_dwi.bval" ] && \
      ln -sf "${subj_folder}_AP_P_dwi.bval" "${subj_folder}_PA_dwi.bval"
    [ -f "${subj_folder}_AP_P_dwi.bvec" ] && \
      ln -sf "${subj_folder}_AP_P_dwi.bvec" "${subj_folder}_PA_dwi.bvec"
    [ -f "${subj_folder}_AP_P_dwi.json" ] && \
      ln -sf "${subj_folder}_AP_P_dwi.json" "${subj_folder}_PA_dwi.json"
  fi

  cd "$BASE_DIR"
done

# Process subjects (single subject or all)
for subj_folder in "${SUBJECT_LIST[@]}"; do
  [ -d "$subj_folder" ] || continue
  echo "Processing subject folder: $subj_folder"
  # --- Clean previous outputs so re-runs don't fail ---
  SUBJ_PREFIX="$subj_folder" 
  cd "$BASE_DIR/$subj_folder" || { echo "Cannot cd to $subj_folder"; cd "$BASE_DIR"; continue; }

  # Clean intermediate files but preserve input DWI/T1 files
  # Only clean if we're doing a fresh run (check if den_preproc_unbiased exists - if it does, this is a re-run)
  # This prevents deleting files if the script failed partway through
  if [ -f "${SUBJ_PREFIX}_den_preproc_unbiased.mif" ]; then
    echo "Found existing preprocessed data - cleaning intermediate files for fresh run..."
    rm -f noise.mif ${SUBJ_PREFIX}_den*.mif mask*.mif mean_b0* \
          bvecs.txt bvals.txt index.txt acqparams.txt \
          topup_results* eddy_corrected* dwifslpreproc-tmp-* \
          5tt_* gmwmSeed_coreg.mif tensor.mif \
          ${SUBJ_PREFIX}_filtered*.tck ${SUBJ_PREFIX}_tracks_10M.tck \
          ${SUBJ_PREFIX}_sift_*.* -r 2>/dev/null
  else
    echo "No existing preprocessed data found - keeping any intermediate files from previous failed run..."
    # Only clean files that are safe to remove (not intermediate denoised files that might be needed)
    rm -f topup_results* eddy_corrected* dwifslpreproc-tmp-* \
          5tt_* gmwmSeed_coreg.mif tensor.mif \
          ${SUBJ_PREFIX}_filtered*.tck ${SUBJ_PREFIX}_tracks_10M.tck \
          ${SUBJ_PREFIX}_sift_*.* -r 2>/dev/null
  fi
  # Remove intermediate .nii files but preserve input DWI/T1 files
  find . -maxdepth 1 -name "${SUBJ_PREFIX}_*.nii" ! -name "${SUBJ_PREFIX}_*_dwi.nii" ! -name "${SUBJ_PREFIX}_*_b0.nii" ! -name "${SUBJ_PREFIX}_T1w.nii" -delete 2>/dev/null || true
  # Check for both .nii.gz and .nii extensions (symlinks may point to either)
  AP_DWI_NII=""
  if [[ -f "$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_AP_dwi.nii.gz" ]]; then
    AP_DWI_NII="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_AP_dwi.nii.gz"
  elif [[ -f "$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_AP_dwi.nii" ]]; then
    AP_DWI_NII="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_AP_dwi.nii"
  fi
  AP_DWI_BVAL="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_AP_dwi.bval"
  AP_DWI_BVEC="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_AP_dwi.bvec"

  PA_DWI_NII=""
  if [[ -f "$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_PA_dwi.nii.gz" ]]; then
    PA_DWI_NII="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_PA_dwi.nii.gz"
  elif [[ -f "$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_PA_dwi.nii" ]]; then
    PA_DWI_NII="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_PA_dwi.nii"
  fi
  PA_DWI_BVAL="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_PA_dwi.bval"
  PA_DWI_BVEC="$BASE_DIR/$subj_folder/${SUBJ_PREFIX}_PA_dwi.bvec"

  MISSING_AP=false
  MISSING_PA=false

  if [[ -z "$AP_DWI_NII" || ! -f "$AP_DWI_BVAL" || ! -f "$AP_DWI_BVEC" ]]; then
    echo "WARNING: AP DWI files not found for $subj_folder."
    if [[ -z "$AP_DWI_NII" ]]; then
      echo "         Missing: ${SUBJ_PREFIX}_AP_dwi.nii.gz or ${SUBJ_PREFIX}_AP_dwi.nii"
    fi
    if [[ ! -f "$AP_DWI_BVAL" ]]; then
      echo "         Missing: ${SUBJ_PREFIX}_AP_dwi.bval"
    fi
    if [[ ! -f "$AP_DWI_BVEC" ]]; then
      echo "         Missing: ${SUBJ_PREFIX}_AP_dwi.bvec"
    fi
    MISSING_AP=true
  fi

  if [[ -z "$PA_DWI_NII" || ! -f "$PA_DWI_BVAL" || ! -f "$PA_DWI_BVEC" ]]; then
    echo "WARNING: PA DWI files not found for $subj_folder."
    if [[ -z "$PA_DWI_NII" ]]; then
      echo "         Missing: ${SUBJ_PREFIX}_PA_dwi.nii.gz or ${SUBJ_PREFIX}_PA_dwi.nii"
    fi
    if [[ ! -f "$PA_DWI_BVAL" ]]; then
      echo "         Missing: ${SUBJ_PREFIX}_PA_dwi.bval"
    fi
    if [[ ! -f "$PA_DWI_BVEC" ]]; then
      echo "         Missing: ${SUBJ_PREFIX}_PA_dwi.bvec"
    fi
    MISSING_PA=true
  fi

  if $MISSING_AP && $MISSING_PA; then
    echo "ERROR: Both AP and PA DWI files are missing for $subj_folder."
    echo "       The DWI NIfTI image files (.nii.gz or .nii) are required for processing."
    echo "       Check if conversion from DICOM completed successfully, or if DICOM data exists."
    echo "Skipping subject $subj_folder (no DWI data available)."
    cd "$BASE_DIR" || exit
    continue
  fi

  if ! $MISSING_AP; then
    DWI_NII="$AP_DWI_NII"
    DWI_BVAL="$AP_DWI_BVAL"
    DWI_BVEC="$AP_DWI_BVEC"
    DWI_PREFIX="${SUBJ_PREFIX}_AP_dwi"
  elif ! $MISSING_PA; then
    DWI_NII="$PA_DWI_NII"
    DWI_BVAL="$PA_DWI_BVAL"
    DWI_BVEC="$PA_DWI_BVEC"
    DWI_PREFIX="${SUBJ_PREFIX}_PA_dwi"
  fi

  cd "$BASE_DIR/$subj_folder" || { echo "Cannot cd to $subj_folder"; cd "$BASE_DIR"; continue; }

  echo "Converting diffusion data to .mif"
  mrconvert "$DWI_NII" "${SUBJ_PREFIX}_dwi.mif" -fslgrad "$DWI_BVEC" "$DWI_BVAL" -force

  if ! $MISSING_AP; then
    cp "$AP_DWI_BVAL" "${SUBJ_PREFIX}_AP.bval"
    cp "$AP_DWI_BVEC" "${SUBJ_PREFIX}_AP.bvec"
  fi
  if ! $MISSING_PA; then
    cp "$PA_DWI_BVAL" "${SUBJ_PREFIX}_PA.bval"
    cp "$PA_DWI_BVEC" "${SUBJ_PREFIX}_PA.bvec"
  fi

  echo "Denoising..."
  dwidenoise "${SUBJ_PREFIX}_dwi.mif" "${SUBJ_PREFIX}_den.mif" -noise noise.mif

  echo "Removing Gibbs ringing..."
  mrdegibbs "${SUBJ_PREFIX}_den.mif" "${SUBJ_PREFIX}_den_unr.mif"
   # -------- PATCH B' (PA DWI + AP b0 -> synth SE-EPI pair) -----------------
  RAW_SUBJ="/media/RCPNAS/Data/Delirium/Delirium_Rania/rawdata_current/$SUBJ_PREFIX/dwi"
  AP_B0_RAW="$RAW_SUBJ/${SUBJ_PREFIX}_dir-AP_b0.nii.gz"
  PA_DWI_RAW="$RAW_SUBJ/${SUBJ_PREFIX}_dir-PA_dwi.nii.gz"
  PA_JSON="$RAW_SUBJ/${SUBJ_PREFIX}_dir-PA_dwi.json"

  TRT=""
  if command -v jq >/dev/null 2>&1 && [[ -f "$PA_JSON" ]]; then
    TRT=$(jq -r '.TotalReadoutTime // .EffectiveEchoSpacing as $e
                 | .ReconMatrixPE as $n
                 | if .TotalReadoutTime then .TotalReadoutTime
                   elif ($e!=null and $n!=null) then ($e*($n-1))
                   else empty end' "$PA_JSON")
  fi

  if [[ -f "$PA_DWI_RAW" && -f "$AP_B0_RAW" ]]; then
    echo "Detected PA DWI + AP b0 for $SUBJ_PREFIX — building SE-EPI b0 pair."
    mrconvert "$PA_DWI_RAW" dwi_input.mif \
      -fslgrad "$RAW_SUBJ/${SUBJ_PREFIX}_dir-PA_dwi.bvec" \
               "$RAW_SUBJ/${SUBJ_PREFIX}_dir-PA_dwi.bval" -force
    dwiextract dwi_input.mif -bzero "${SUBJ_PREFIX}_PA_b0s.mif"
    mrmath "${SUBJ_PREFIX}_PA_b0s.mif" mean mean_b0_PA.mif -axis 3
    mrconvert mean_b0_PA.mif mean_b0_PA.nii.gz -force
    cp "$AP_B0_RAW" AP_b0_raw.nii.gz
    mrgrid AP_b0_raw.nii.gz regrid -template mean_b0_PA.nii.gz mean_b0_AP.nii.gz
    cp mean_b0_AP.nii.gz "${SUBJ_PREFIX}_dir-AP_epi.nii.gz"
    cp mean_b0_PA.nii.gz "${SUBJ_PREFIX}_dir-PA_epi.nii.gz"
  fi


  # -------- PATCH B (prefer dwifslpreproc if SE-EPI fmaps exist) -----------
  FWD_EPI="${SUBJ_PREFIX}_dir-AP_epi.nii.gz"
  REV_EPI="${SUBJ_PREFIX}_dir-PA_epi.nii.gz"
  if [[ -f "$FWD_EPI" && -f "$REV_EPI" ]]; then
    echo "Found SE-EPI fieldmaps — using dwifslpreproc (rpe_pair)."
    mrconvert "$DWI_NII" dwi_input.mif -fslgrad "$DWI_BVEC" "$DWI_BVAL" -force
     fslmerge -t mean_b0_pair.nii.gz \
      "${SUBJ_PREFIX}_dir-AP_epi.nii.gz" "${SUBJ_PREFIX}_dir-PA_epi.nii.gz"

    dwifslpreproc dwi_input.mif "${SUBJ_PREFIX}_den_preproc.mif" \
      -rpe_pair -se_epi mean_b0_pair.nii.gz -pe_dir PA \
      -eddy_options " --slm=linear " -nocleanup -force  

    # Export merged grads for later steps
    mrinfo "${SUBJ_PREFIX}_den_preproc.mif" -export_grad_fsl bvecs.txt bvals.txt

  elif ! $MISSING_AP && ! $MISSING_PA; then
    # -------------------- ORIGINAL MANUAL TOPUP+EDDY BLOCK ------------------
    echo "Running manual topup + eddy distortion correction..."

    mrconvert "$AP_DWI_NII" "${SUBJ_PREFIX}_AP_dwi_new.mif" -fslgrad "$AP_DWI_BVEC" "$AP_DWI_BVAL"
    mrconvert "$PA_DWI_NII" "${SUBJ_PREFIX}_PA_dwi_new.mif" -fslgrad "$PA_DWI_BVEC" "$PA_DWI_BVAL"

    mrcat "${SUBJ_PREFIX}_AP_dwi_new.mif" "${SUBJ_PREFIX}_PA_dwi_new.mif" -axis 3 "${SUBJ_PREFIX}_merged_dwi.mif"

    (tr -d '\n' < "${SUBJ_PREFIX}_AP.bval"; echo -n " "; tr -d '\n' < "${SUBJ_PREFIX}_PA.bval"; echo) | awk '{$1=$1; print}' > bvals.txt
    paste -d ' ' <(awk 'NR==1' "${SUBJ_PREFIX}_AP.bvec") <(awk 'NR==1' "${SUBJ_PREFIX}_PA.bvec") > bvecs_x.txt
    paste -d ' ' <(awk 'NR==2' "${SUBJ_PREFIX}_AP.bvec") <(awk 'NR==2' "${SUBJ_PREFIX}_PA.bvec") > bvecs_y.txt
    paste -d ' ' <(awk 'NR==3' "${SUBJ_PREFIX}_AP.bvec") <(awk 'NR==3' "${SUBJ_PREFIX}_PA.bvec") > bvecs_z.txt
    cat bvecs_x.txt bvecs_y.txt bvecs_z.txt > bvecs.txt
    rm bvecs_x.txt bvecs_y.txt bvecs_z.txt

    # -------- PATCH A usage (auto acqparams from JSON) ---------------------
    AP_JSON="${SUBJ_PREFIX}_AP_dwi.json"
    PA_JSON="${SUBJ_PREFIX}_PA_dwi.json"
    TRT_AP=""; TRT_PA=""
    [[ -f "$AP_JSON" ]] && TRT_AP=$(read_trt_from_json "$AP_JSON")
    [[ -f "$PA_JSON" ]] && TRT_PA=$(read_trt_from_json "$PA_JSON")
    TRT="${TRT_AP:-${TRT_PA:-0.0450000}}"

    cat > acqparams.txt <<EOF
0 -1 0 ${TRT}
0  1 0 ${TRT}
EOF
    # -----------------------------------------------------------------------

    # Export merged DWI to both .mif and .nii.gz formats
    mrconvert "${SUBJ_PREFIX}_merged_dwi.mif" dwi_merged.mif \
      -export_grad_fsl bvecs.txt bvals.txt -force

    mrconvert "${SUBJ_PREFIX}_merged_dwi.mif" dwi_merged.nii.gz \
      -export_grad_fsl bvecs.txt bvals.txt -force

    dwiextract "${SUBJ_PREFIX}_AP_dwi_new.mif" -bzero "${SUBJ_PREFIX}_AP_b0s.mif"
    dwiextract "${SUBJ_PREFIX}_PA_dwi_new.mif" -bzero "${SUBJ_PREFIX}_PA_b0s.mif"

    mrmath "${SUBJ_PREFIX}_AP_b0s.mif" mean mean_b0_AP.mif -axis 3
    mrmath "${SUBJ_PREFIX}_PA_b0s.mif" mean mean_b0_PA.mif -axis 3

    mrconvert mean_b0_AP.mif mean_b0_AP.nii.gz
    mrconvert mean_b0_PA.mif mean_b0_PA.nii.gz
    fslmerge -t mean_b0_pair.nii.gz mean_b0_AP.nii.gz mean_b0_PA.nii.gz

    topup --imain=mean_b0_pair.nii.gz --datain=acqparams.txt --config=b02b0.cnf --out=topup_results --iout=topup_corrected_b0.nii.gz

    N_AP=$(mrinfo "${SUBJ_PREFIX}_AP_dwi_new.mif" -size | awk '{print $4}')
    N_PA=$(mrinfo "${SUBJ_PREFIX}_PA_dwi_new.mif" -size | awk '{print $4}')
    index_file=""
    for ((i=1; i<=N_AP; i++)); do index_file+="1 "; done
    for ((i=1; i<=N_PA; i++)); do index_file+="2 "; done
    echo "$index_file" > index.txt

    echo "Generating brain mask for eddy..."
    # Generate mask from merged DWI (after topup b0 correction, before eddy)
    dwi2mask dwi_merged.mif mask.mif -clean_scale 1
    # Convert brain mask
    mrconvert mask.mif mask.nii.gz

    eddy \
      --imain=dwi_merged.nii.gz \
      --mask=mask.nii.gz \
      --acqp=acqparams.txt \
      --index=index.txt \
      --bvecs=bvecs.txt \
      --bvals=bvals.txt \
      --topup=topup_results \
      --out=eddy_corrected \
      --verbose
    if [ ! -f eddy_corrected.nii.gz ]; then
        echo "ERROR: Eddy failed for $SUBJ_PREFIX — skipping subject."
        cd "$BASE_DIR"
        continue
    fi
    mrconvert eddy_corrected.nii.gz "${SUBJ_PREFIX}_den_preproc.mif"  -fslgrad bvecs.txt bvals.txt -force

  else
    # -------- Single PE direction: eddy-only with T1-based distortion correction (if available) ----
    echo "Only one phase encoding direction present — running eddy with T1-based distortion correction (if available)."
    
    # Determine which DWI we're using and get PE direction
    DWI_JSON=""
    PE_LABEL=""
    if ! $MISSING_AP; then
      DWI_JSON="${SUBJ_PREFIX}_AP_dwi.json"
      PE_LABEL="AP"
    elif ! $MISSING_PA; then
      DWI_JSON="${SUBJ_PREFIX}_PA_dwi.json"
      PE_LABEL="PA"
    fi
    
    # Check if T1 is available for distortion correction
    T1_NII_FOR_DIST=""
    if [ -f "${SUBJ_PREFIX}_T1w.nii.gz" ]; then
      T1_NII_FOR_DIST="${SUBJ_PREFIX}_T1w.nii.gz"
    elif [ -f "${SUBJ_PREFIX}_T1w.nii" ]; then
      T1_NII_FOR_DIST="${SUBJ_PREFIX}_T1w.nii"
    fi
    
    # Get PE direction for dwifslpreproc
    PE_DIR=""
    if [[ -n "$DWI_JSON" ]] && [[ -f "$DWI_JSON" ]]; then
      if command -v jq >/dev/null 2>&1; then
        PE_DIR=$(jq -r '.PhaseEncodingDirection // empty' "$DWI_JSON" 2>/dev/null)
      fi
    fi
    
    # For single-PE cases, use eddy-only (motion/eddy current correction only, no fieldmap correction)
    # Note: dwifslpreproc with -rpe_none doesn't do distortion correction, so we use eddy directly
    # with a DWI-based mask (using -clean_scale 1) for best results
    echo "Running eddy-only (motion/eddy current correction, no fieldmap correction)..."
    
    # Get TRT and PE vector from JSON
    TRT="0.0450000"
    if [[ -n "$DWI_JSON" ]] && [[ -f "$DWI_JSON" ]]; then
      TRT=$(read_trt_from_json "$DWI_JSON")
    fi
    
    PE_VECTOR=$(get_pe_vector_from_json "$DWI_JSON" "$PE_LABEL")
    
    # Create acqparams.txt for single PE direction
    cat > acqparams.txt <<EOF
${PE_VECTOR} ${TRT}
EOF
    
    # Convert denoised/unringed DWI to NIfTI format for eddy
    echo "Converting to NIfTI format for eddy..."
    if [[ -f "${SUBJ_PREFIX}_${PE_LABEL}.bvec" ]] && [[ -f "${SUBJ_PREFIX}_${PE_LABEL}.bval" ]]; then
      mrconvert "${SUBJ_PREFIX}_den_unr.mif" dwi_single_pe.nii.gz \
        -export_grad_fsl "${SUBJ_PREFIX}_${PE_LABEL}.bvec" "${SUBJ_PREFIX}_${PE_LABEL}.bval" -force
      cp "${SUBJ_PREFIX}_${PE_LABEL}.bvec" bvecs.txt
      cp "${SUBJ_PREFIX}_${PE_LABEL}.bval" bvals.txt
    else
      if [[ -f "$DWI_BVEC" ]] && [[ -f "$DWI_BVAL" ]]; then
        mrconvert "${SUBJ_PREFIX}_den_unr.mif" dwi_single_pe.nii.gz \
          -export_grad_fsl "$DWI_BVEC" "$DWI_BVAL" -force
        cp "$DWI_BVEC" bvecs.txt
        cp "$DWI_BVAL" bvals.txt
      else
        echo "ERROR: Cannot find bvec/bval files for $SUBJ_PREFIX — skipping eddy correction."
        cp "${SUBJ_PREFIX}_den_unr.mif" "${SUBJ_PREFIX}_den_preproc.mif" -f
        cd "$BASE_DIR" || exit
        continue
      fi
    fi
    
    # Create index.txt
    N_VOL=$(mrinfo "${SUBJ_PREFIX}_den_unr.mif" -size | awk '{print $4}')
    index_file=""
    for ((i=1; i<=N_VOL; i++)); do
      index_file+="1 "
    done
    echo "$index_file" > index.txt
    
    # Generate brain mask for eddy using DWI-based mask (proven best approach)
    echo "Generating brain mask for eddy..."
    dwi2mask "${SUBJ_PREFIX}_den_unr.mif" mask.mif -clean_scale 1
    mrconvert mask.mif mask.nii.gz -force
    
    # Run eddy without topup (motion and eddy current correction only)
    echo "Running eddy (motion/eddy currents corrected, fieldmap correction skipped)..."
    eddy \
      --imain=dwi_single_pe.nii.gz \
      --mask=mask.nii.gz \
      --acqp=acqparams.txt \
      --index=index.txt \
      --bvecs=bvecs.txt \
      --bvals=bvals.txt \
      --out=eddy_corrected \
      --verbose
    
    if [ ! -f eddy_corrected.nii.gz ]; then
      echo "WARNING: Eddy failed for $SUBJ_PREFIX — using denoised/unringed data without eddy correction."
      cp "${SUBJ_PREFIX}_den_unr.mif" "${SUBJ_PREFIX}_den_preproc.mif" -f
    else
      # Convert eddy output back to .mif format
      mrconvert eddy_corrected.nii.gz "${SUBJ_PREFIX}_den_preproc.mif" \
        -fslgrad bvecs.txt bvals.txt -force
      echo "Eddy correction completed (motion/eddy currents corrected, fieldmap correction skipped)."
    fi
    # -------------------------------------------------------------------------
  fi
  # -------- END PATCH B -----------------------------------------------------

  # Verify that den_preproc.mif was created successfully
  if [ ! -f "${SUBJ_PREFIX}_den_preproc.mif" ]; then
    echo "ERROR: ${SUBJ_PREFIX}_den_preproc.mif not found. Preprocessing (eddy/dwifslpreproc) may have failed."
    echo "       Skipping subject $subj_folder."
    cd "$BASE_DIR" || exit
    continue
  fi

  echo "Generating brain mask for bias correction..."
  dwi2mask "${SUBJ_PREFIX}_den_preproc.mif" mask_prebias.mif -clean_scale 1
  
  echo "Running bias field correction..."
  if command -v N4BiasFieldCorrection >/dev/null 2>&1; then
    echo "Using ANTs algorithm for bias correction..."
    dwibiascorrect ants "${SUBJ_PREFIX}_den_preproc.mif" "${SUBJ_PREFIX}_den_preproc_unbiased.mif" \
      -mask mask_prebias.mif -bias bias.mif -force
  else
    echo "ANTs not available, using FSL for bias correction..."
    dwibiascorrect fsl "${SUBJ_PREFIX}_den_preproc.mif" "${SUBJ_PREFIX}_den_preproc_unbiased.mif" \
      -mask mask_prebias.mif -bias bias.mif -force
  fi
  
  # Verify bias correction completed successfully
  if [ ! -f "${SUBJ_PREFIX}_den_preproc_unbiased.mif" ]; then
    echo "ERROR: Bias correction failed - ${SUBJ_PREFIX}_den_preproc_unbiased.mif not created."
    echo "       Using den_preproc.mif without bias correction as fallback."
    cp "${SUBJ_PREFIX}_den_preproc.mif" "${SUBJ_PREFIX}_den_preproc_unbiased.mif" -f
  fi

  echo "Generating brain mask from bias-corrected data..."
  dwi2mask "${SUBJ_PREFIX}_den_preproc_unbiased.mif" mask_unbiased.mif -clean_scale 1

  echo "Estimating response functions (WM And Single Shell)"
  dwi2response tournier "${SUBJ_PREFIX}_den_preproc_unbiased.mif" \
    "${SUBJ_PREFIX}_wm_response.txt" \
     -mask mask_unbiased.mif -force

  echo "Estimating WM FODs (single-shell CSD)…"
  dwi2fod csd "${SUBJ_PREFIX}_den_preproc_unbiased.mif" \
    "${SUBJ_PREFIX}_wm_response.txt" \
    "${SUBJ_PREFIX}_wmfod.mif" \
    -mask mask_unbiased.mif -force

  echo "Normalizing WM FODs..."
  mtnormalise "${SUBJ_PREFIX}_wmfod.mif" "${SUBJ_PREFIX}_wmfod_norm.mif" \
  -mask mask_unbiased.mif


  echo "Converting anatomical images to .mif..."
  # Check for both .nii.gz and .nii extensions
  T1_NII=""
  if [ -f "${SUBJ_PREFIX}_T1w.nii.gz" ]; then
    T1_NII="${SUBJ_PREFIX}_T1w.nii.gz"
  elif [ -f "${SUBJ_PREFIX}_T1w.nii" ]; then
    T1_NII="${SUBJ_PREFIX}_T1w.nii"
  fi
  
  MISSING_T1=false
  if [ -n "$T1_NII" ] && [ -f "$T1_NII" ]; then
    mrconvert "$T1_NII" T1.mif -force
    echo "Generating 5tt image..."
    5ttgen fsl T1.mif 5tt_nocoreg.mif
    
    echo "Extracting and averaging b=0 volumes to create mean_b0.mif..."
    dwiextract "${SUBJ_PREFIX}_den_preproc_unbiased.mif" - -bzero | mrmath - mean mean_b0.mif -axis 3 -force
    if [ ! -f mean_b0.mif ]; then
      echo "ERROR: mean_b0.mif not created. Exiting."
      exit 1
    fi
    
    echo "Coregistering DWI to structural..."
    mrconvert mean_b0.mif mean_b0.nii.gz -force
    mrconvert 5tt_nocoreg.mif 5tt_nocoreg.nii.gz -force
    fslroi 5tt_nocoreg.nii.gz 5tt_vol0.nii.gz 0 1
    flirt -in mean_b0.nii.gz -ref 5tt_vol0.nii.gz -interp nearestneighbour -dof 6 -omat diff2struct_fsl.mat
    transformconvert diff2struct_fsl.mat mean_b0.nii.gz 5tt_nocoreg.nii.gz flirt_import diff2struct_mrtrix.txt
    mrtransform 5tt_nocoreg.mif -linear diff2struct_mrtrix.txt -inverse 5tt_coreg.mif
    5tt2gmwmi 5tt_coreg.mif gmwmSeed_coreg.mif
  else
    echo "WARNING: T1w image not found for $subj_folder - skipping T1-dependent steps (coregistration, ACT-based tractography)."
    MISSING_T1=true
    # Create dummy files to prevent errors in later steps
    echo "Creating dummy 5tt and seed files for DWI-only processing..."
    dwiextract "${SUBJ_PREFIX}_den_preproc_unbiased.mif" - -bzero | mrmath - mean mean_b0.mif -axis 3 -force
    # Create a simple mask-based seed instead of gmwmSeed
    dwi2mask "${SUBJ_PREFIX}_den_preproc_unbiased.mif" gmwmSeed_coreg.mif -clean_scale 1 -force
  fi

  echo "Fitting tensor..."
  dwi2tensor -mask mask_unbiased.mif "${SUBJ_PREFIX}_den_preproc_unbiased.mif" tensor.mif

  echo "Creating DTI scalar maps..."
  tensor2metric tensor.mif -fa "${SUBJ_PREFIX}_FA.nii" -adc "${SUBJ_PREFIX}_MD.nii" -ad "${SUBJ_PREFIX}_AD.nii" -rd "${SUBJ_PREFIX}_RD.nii"

  echo "Generating tractogram..."
  if ! $MISSING_T1; then
    echo "Using ACT-based tractography (with T1)..."
    tckgen "${SUBJ_PREFIX}_wmfod_norm.mif" "${SUBJ_PREFIX}_tracks_10M.tck" \
      -act 5tt_coreg.mif -backtrack -seed_gmwmi gmwmSeed_coreg.mif \
      -nthreads 8 -maxlength 250 -cutoff 0.06 -select 10000000
    
    echo "Filtering with SIFT2..."
    tcksift2 "${SUBJ_PREFIX}_tracks_10M.tck" "${SUBJ_PREFIX}_wmfod_norm.mif" \
      "${SUBJ_PREFIX}_sift_1M.txt" -act 5tt_coreg.mif \
      -out_mu "${SUBJ_PREFIX}_sift_mu.txt" \
      -out_coeffs "${SUBJ_PREFIX}_sift_coeffs.txt" -nthreads 8
  else
    echo "Using standard tractography (without T1, no ACT)..."
    tckgen "${SUBJ_PREFIX}_wmfod_norm.mif" "${SUBJ_PREFIX}_tracks_10M.tck" \
      -seed_image gmwmSeed_coreg.mif -mask mask_unbiased.mif \
      -nthreads 8 -maxlength 250 -cutoff 0.06 -select 10000000
    
    echo "Filtering with SIFT2 (without ACT)..."
    tcksift2 "${SUBJ_PREFIX}_tracks_10M.tck" "${SUBJ_PREFIX}_wmfod_norm.mif" \
      "${SUBJ_PREFIX}_sift_1M.txt" \
      -out_mu "${SUBJ_PREFIX}_sift_mu.txt" \
      -out_coeffs "${SUBJ_PREFIX}_sift_coeffs.txt" -nthreads 8
  fi

  tckedit "${SUBJ_PREFIX}_tracks_10M.tck" \
    "${SUBJ_PREFIX}_filtered.tck" \
    -tck_weights_in "${SUBJ_PREFIX}_sift_1M.txt" \
    -minweight 0.001

  cd "$BASE_DIR" || exit
done

echo "All subjects processed."
