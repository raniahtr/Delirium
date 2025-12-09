#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# DICOM -> NIfTI (+JSON/bval/bvec) with dcm2niix
# Subject ID normalization (remove trailing underscores)
# Outputs:
#   - BIDS-like "rawdata/"
#   - Preproc/"sub-XX/" (flat; NO 00_raw, NO Files)
# Includes:
#   - DWI main (+ reverse-PE b0 if present)
#   - T1 (anat) also copied into Preproc
#   - FMAP (SE-EPI) -> rawdata/sub-XX/fmap + a copy into Preproc/sub-XX
# Robust PED inference:
#   - JSON PhaseEncodingDirection if present
#   - DICOM-derived fields via pe_json_fix.py
#   - Reverse-PE b0 series (opposite polarity) if needed
# Leaves Original_Data/ untouched.
# 
# Usage:
#   ./convert_and_organize.sh                    # Process all subjects
#   ONLY_NEW_FILES=1 ./convert_and_organize.sh   # Process only 'sub-*' directories
# ----------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

IN_ROOT="${IN_ROOT:-${SCRIPT_DIR}/../../Original_Data}"
OUT_RAW="${OUT_RAW:-${SCRIPT_DIR}/../rawdata_current}"
OUT_PRE="${OUT_PRE:-${SCRIPT_DIR}/../Preproc_current}"
TMP_DIR="${TMP_DIR:-${SCRIPT_DIR}/../_dcm2niix_tmp}"
DCM2NIIX="${DCM2NIIX:-dcm2niix}"
# If set to 1, only process directories starting with 'sub-' (new data format)
ONLY_NEW_FILES="${ONLY_NEW_FILES:-0}"

JSON_FIXER="${SCRIPT_DIR}/pe_json_fix.py"

mkdir -p "$OUT_RAW" "$OUT_PRE" "$TMP_DIR"

log(){ echo "[Convert] $*"; }
warn(){ echo "[WARN] $*" >&2; }

command -v "$DCM2NIIX" >/dev/null 2>&1 || { echo "dcm2niix not found in PATH"; exit 1; }
if ! command -v jq >/dev/null 2>&1; then
  warn "jq not found; JSON checks/patches limited (script still runs)."
fi
[[ -f "$JSON_FIXER" ]] || warn "JSON fixer not found at: $JSON_FIXER (continuing without it)."

# ---- helpers ----
norm_subid() {
  local raw="$1"
  local s="${raw%%_}"
  s="${s%%_}"
  # Strip 'sub-' prefix if present (to avoid double prefix like sub-subAR)
  s="${s#sub-}"
  s="${s#sub}"
  # Remove all non-alphanumeric characters
  s="${s//[^A-Za-z0-9]/}"
  printf "%s" "$s"
}

install_bids() {
  local sub="$1" kind="$2" src_base="$3"
  local dst_dir
  case "$kind" in
    dwi)  dst_dir="${OUT_RAW}/sub-${sub}/dwi" ;;
    anat) dst_dir="${OUT_RAW}/sub-${sub}/anat" ;;
    fmap) dst_dir="${OUT_RAW}/sub-${sub}/fmap" ;;
    *)    dst_dir="${OUT_RAW}/sub-${sub}/${kind}" ;;
  esac
  mkdir -p "$dst_dir"
  shopt -s nullglob
  # Copy .nii.gz first (preferred), then .nii if .nii.gz doesn't exist
  if [[ -f "${src_base}.nii.gz" ]]; then
    cp -f "${src_base}.nii.gz" "${dst_dir}/"
  elif [[ -f "${src_base}.nii" ]]; then
    cp -f "${src_base}.nii" "${dst_dir}/"
  fi
  # Copy other associated files
  for ext in json bval bvec; do
    [[ -f "${src_base}.${ext}" ]] && cp -f "${src_base}.${ext}" "${dst_dir}/"
  done
  shopt -u nullglob
}

install_preproc() {
  local sub="$1" src_base="$2" stem="$3"
  local dst_dir="${OUT_PRE}/sub-${sub}"
  mkdir -p "$dst_dir"
  shopt -s nullglob
  # Copy .nii.gz first (preferred), then .nii if .nii.gz doesn't exist
  if [[ -f "${src_base}.nii.gz" ]]; then
    cp -f "${src_base}.nii.gz" "${dst_dir}/${stem}.nii.gz"
  elif [[ -f "${src_base}.nii" ]]; then
    cp -f "${src_base}.nii" "${dst_dir}/${stem}.nii"
  fi
  # Copy other associated files
  for ext in json bval bvec; do
    [[ -f "${src_base}.${ext}" ]] && cp -f "${src_base}.${ext}" "${dst_dir}/${stem}.${ext}"
  done
  shopt -u nullglob
}

# Strip .nii.gz or .nii extension to get base filename
strip_nii_ext() {
  local file="$1"
  file="${file%.nii.gz}"
  file="${file%.nii}"
  printf "%s" "$file"
}

pe_to_preproc_tag() {
  case "$1" in
    j)  echo "AP_A";;  j-) echo "AP_P";;
    i)  echo "LR_L";;  i-) echo "LR_R";;
    k)  echo "FH_F";;  k-) echo "FH_H";;
    *)  echo "AP_A";;
  esac
}

opposite_ped() {
  case "$1" in
    j)  echo "j-";; j-) echo "j";;
    i)  echo "i-";; i-) echo "i";;
    k)  echo "k-";; k-) echo "k";;
    *)  echo "";;
  esac
}

read_ped() {
  local json="$1"
  if command -v jq >/dev/null 2>&1 && [[ -f "$json" ]]; then
    jq -r '.PhaseEncodingDirection // empty' "$json" 2>/dev/null || true
  else
    echo ""
  fi
}

add_intendedfor() {
  local json="$1"; shift
  if command -v jq >/dev/null 2>&1 && [[ -f "$json" ]]; then
    local tmp="${json}.tmp"
    # Merge unique paths into IntendedFor array
    jq --argjson add "$(printf '%s\n' "$@" | jq -R . | jq -s '.')" '
      .IntendedFor = (
        ( .IntendedFor // [] )
        + $add
      ) | .IntendedFor |= (unique)
    ' "$json" > "$tmp" && mv "$tmp" "$json"
  fi
}

# ---- main ----
for subj_dir in "${IN_ROOT}"/*; do
  [[ -d "$subj_dir" ]] || continue
  raw_id="$(basename "$subj_dir")"
  
  # Skip if ONLY_NEW_FILES is enabled and directory doesn't start with 'sub-'
  if [[ "$ONLY_NEW_FILES" == "1" ]] && [[ ! "$raw_id" =~ ^sub- ]]; then
    log "Skipping ${raw_id} (not a 'sub-*' directory, use ONLY_NEW_FILES=0 to process all)"
    continue
  fi
  
  sub="$(norm_subid "$raw_id")"
  log "Subject: ${raw_id}  -> normalized: ${sub}"

  sub_tmp="${TMP_DIR}/${sub}"
  rm -rf "$sub_tmp" && mkdir -p "$sub_tmp"

  # Process each series (handle both DICOM and already-converted NIfTI)
  for series in "$subj_dir"/*; do
    [[ -d "$series" ]] || continue
    series_name="$(basename "$series")"
    
    # Check if directory contains NIfTI files (already converted)
    has_nifti=false
    if find "$series" -maxdepth 1 -type f \( -name "*.nii" -o -name "*.nii.gz" \) 2>/dev/null | head -1 | grep -q .; then
      has_nifti=true
    fi
    
    # Check if directory contains DICOM files
    has_dicom=false
    # Check for DICOM with extensions
    if find "$series" -maxdepth 2 -type f \( -name "*.dcm" -o -name "DICOMDIR" -o -name "*.DCM" \) 2>/dev/null | head -1 | grep -q .; then
      has_dicom=true
    else
      # Check for extensionless files that might be DICOM
      file_count=$(find "$series" -maxdepth 1 -type f ! -name ".*" ! -name "*.nii" ! -name "*.nii.gz" ! -name "*.json" ! -name "*.bval" ! -name "*.bvec" ! -name "*.txt" ! -name "*.log" 2>/dev/null | wc -l)
      if [[ "$file_count" -gt 0 ]]; then
        # Sample a file to check if it's DICOM
        sample_file=$(find "$series" -maxdepth 1 -type f ! -name ".*" ! -name "*.nii" ! -name "*.nii.gz" ! -name "*.json" ! -name "*.bval" ! -name "*.bvec" 2>/dev/null | head -1)
        if [[ -n "$sample_file" ]] && command -v file >/dev/null 2>&1; then
          if file "$sample_file" 2>/dev/null | grep -qi "DICOM"; then
            has_dicom=true
          fi
        elif [[ -n "$sample_file" ]]; then
          # If 'file' command not available, assume extensionless files in series dirs are DICOM
          has_dicom=true
        fi
      fi
    fi
    
    if [[ "$has_nifti" == "true" ]]; then
      # Copy already-converted NIfTI files
      log "  Copying NIfTI files: ${series_name}"
      shopt -s nullglob
      for nii_file in "$series"/*.{nii,nii.gz,json,bval,bvec} "$series"/*.{NII,NII.GZ,JSON,BVAL,BVEC}; do
        [[ -f "$nii_file" ]] && cp -f "$nii_file" "$sub_tmp/"
      done
      shopt -u nullglob
    elif [[ "$has_dicom" == "true" ]]; then
      # Convert DICOM to NIfTI
      log "  dcm2niix: ${series_name}"
      "$DCM2NIIX" -z y -f "%p_%s" -o "$sub_tmp" "$series" >/dev/null 2>&1 || warn "    dcm2niix failed for ${series_name}"
    else
      warn "  No DICOM or NIfTI files found in: ${series_name}"
    fi
  done

  # Identify series (handle both .nii and .nii.gz, flexible patterns)
  shopt -s nullglob nocaseglob
  # DWI patterns: match diff/MDDW/DTI files (with or without "64" in name, handle "68 directions" too)
  dwi_main=( "$sub_tmp"/*diff*64*.nii.gz "$sub_tmp"/*MDDW*64*.nii.gz \
             "$sub_tmp"/*diff*64*.nii "$sub_tmp"/*MDDW*64*.nii \
             "$sub_tmp"/*DTI*.nii.gz "$sub_tmp"/*DTI*.nii \
             "$sub_tmp"/*diff*.nii.gz "$sub_tmp"/*MDDW*.nii.gz \
             "$sub_tmp"/*diff*.nii "$sub_tmp"/*MDDW*.nii \
             "$sub_tmp"/*ep2d*diff*.nii.gz "$sub_tmp"/*ep2d*diff*.nii )
  # Filter out b0 files from main DWI
  dwi_main_filtered=()
  for dwi_file in "${dwi_main[@]}"; do
    if [[ ! "$dwi_file" =~ [Bb]0 ]] && [[ ! "$dwi_file" =~ _b0_ ]]; then
      dwi_main_filtered+=("$dwi_file")
    fi
  done
  dwi_main=("${dwi_main_filtered[@]}")
  
  dwi_pa_b0=( "$sub_tmp"/*B0*.nii.gz "$sub_tmp"/*_b0_*.nii.gz "$sub_tmp"/*B0*.nii "$sub_tmp"/*_b0_*.nii )
  anat_t1=( "$sub_tmp"/*mprage*.nii.gz "$sub_tmp"/*t1*.nii.gz "$sub_tmp"/*mprage*.nii "$sub_tmp"/*t1*.nii "$sub_tmp"/*T1*.nii.gz "$sub_tmp"/*T1*.nii )
  # common SE-EPI fmap names
  fmap_epi=( "$sub_tmp"/*_epi.nii.gz "$sub_tmp"/*fmap*.nii.gz "$sub_tmp"/*spin*echo*epi*.nii.gz "$sub_tmp"/*_epi.nii "$sub_tmp"/*fmap*.nii "$sub_tmp"/*spin*echo*epi*.nii )
  shopt -u nocaseglob

  # --- DWI main ---
  if (( ${#dwi_main[@]} )); then
    dwi_main_nii="${dwi_main[0]}"; dwi_base="$(strip_nii_ext "$dwi_main_nii")"
    dwi_json="${dwi_base}.json"
    [[ -f "$JSON_FIXER" && -f "$dwi_json" ]] && python3 "$JSON_FIXER" "$dwi_json" || true

    # reverse b0 (optional) early read
    ped_b0=""
    if (( ${#dwi_pa_b0[@]} )); then
      b0_nii="${dwi_pa_b0[0]}"; b0_base="$(strip_nii_ext "$b0_nii")"; b0_json="${b0_base}.json"
      [[ -f "$JSON_FIXER" && -f "$b0_json" ]] && python3 "$JSON_FIXER" "$b0_json" || true
      ped_b0="$(read_ped "$b0_json")"
    fi

    ped="$(read_ped "$dwi_json")"
    if [[ -z "$ped" && -n "$ped_b0" ]]; then ped="$(opposite_ped "$ped_b0")"; fi
    [[ -z "$ped" ]] && { warn "PED missing for ${sub} main DWI; assuming 'j' (AP)."; ped="j"; }

    # BIDS dwi
    install_bids "$sub" "dwi" "$dwi_base"
    if command -v jq >/dev/null 2>&1 && [[ -f "${OUT_RAW}/sub-${sub}/dwi/$(basename "$dwi_base").json" ]]; then
      python3 "$JSON_FIXER" "${OUT_RAW}/sub-${sub}/dwi/$(basename "$dwi_base").json" || true
    fi
    dir_tag="AP"; [[ "$ped" == "j-" ]] && dir_tag="PA"
    pushd "${OUT_RAW}/sub-${sub}/dwi" >/dev/null
      # Handle .nii.gz and .nii files
      if [[ -f "$(basename "$dwi_base").nii.gz" ]]; then
        mv -f "$(basename "$dwi_base").nii.gz" "sub-${sub}_dir-${dir_tag}_dwi.nii.gz"
      elif [[ -f "$(basename "$dwi_base").nii" ]]; then
        mv -f "$(basename "$dwi_base").nii" "sub-${sub}_dir-${dir_tag}_dwi.nii"
      fi
      # Handle other associated files
      for ext in json bval bvec; do
        [[ -f "$(basename "$dwi_base").${ext}" ]] && mv -f "$(basename "$dwi_base").${ext}" "sub-${sub}_dir-${dir_tag}_dwi.${ext}"
      done
    popd >/dev/null

    # Preproc dwi
    pre_tag="$(pe_to_preproc_tag "$ped")"
    install_preproc "$sub" "$dwi_base" "sub-${sub}_${pre_tag}_dwi"
  else
    warn "No main DWI found for ${sub}"
  fi

  # --- Reverse-PE b0 (optional) ---
  if (( ${#dwi_pa_b0[@]} )); then
    b0_nii="${dwi_pa_b0[0]}"; b0_base="$(strip_nii_ext "$b0_nii")"; b0_json="${b0_base}.json"
    install_bids "$sub" "dwi" "$b0_base"
    if command -v jq >/dev/null 2>&1 && [[ -f "${OUT_RAW}/sub-${sub}/dwi/$(basename "$b0_base").json" ]]; then
      python3 "$JSON_FIXER" "${OUT_RAW}/sub-${sub}/dwi/$(basename "$b0_base").json" || true
    fi
    ped_b0_final="$(read_ped "$b0_json")"; [[ -z "$ped_b0_final" ]] && ped_b0_final="j-"
    dir_tag_b0="AP"; [[ "$ped_b0_final" == "j-" ]] && dir_tag_b0="PA"
    pushd "${OUT_RAW}/sub-${sub}/dwi" >/dev/null
      # Handle .nii.gz and .nii files
      if [[ -f "$(basename "$b0_base").nii.gz" ]]; then
        mv -f "$(basename "$b0_base").nii.gz" "sub-${sub}_dir-${dir_tag_b0}_b0.nii.gz"
      elif [[ -f "$(basename "$b0_base").nii" ]]; then
        mv -f "$(basename "$b0_base").nii" "sub-${sub}_dir-${dir_tag_b0}_b0.nii"
      fi
      # Handle JSON
      [[ -f "$(basename "$b0_base").json" ]] && mv -f "$(basename "$b0_base").json" "sub-${sub}_dir-${dir_tag_b0}_b0.json"
    popd >/dev/null

    install_preproc "$sub" "$b0_base" "sub-${sub}_$(pe_to_preproc_tag "$ped_b0_final")_b0"
  fi

  # --- T1 (also into Preproc) ---
  if (( ${#anat_t1[@]} )); then
    t1_nii="${anat_t1[0]}"; t1_base="$(strip_nii_ext "$t1_nii")"; t1_json="${t1_base}.json"
    [[ -f "$JSON_FIXER" && -f "$t1_json" ]] && python3 "$JSON_FIXER" "$t1_json" || true

    install_bids "$sub" "anat" "$t1_base"
    pushd "${OUT_RAW}/sub-${sub}/anat" >/dev/null
      # Handle .nii.gz and .nii files
      if [[ -f "$(basename "$t1_base").nii.gz" ]]; then
        mv -f "$(basename "$t1_base").nii.gz" "sub-${sub}_T1w.nii.gz"
      elif [[ -f "$(basename "$t1_base").nii" ]]; then
        mv -f "$(basename "$t1_base").nii" "sub-${sub}_T1w.nii"
      fi
      # Handle JSON
      [[ -f "$(basename "$t1_base").json" ]] && mv -f "$(basename "$t1_base").json" "sub-${sub}_T1w.json"
    popd >/dev/null

    install_preproc "$sub" "$t1_base" "sub-${sub}_T1w"
  else
    warn "No T1 found for ${sub}"
  fi

  # --- FMAP (SE-EPI) -> BIDS fmap + copy to Preproc ---
  if (( ${#fmap_epi[@]} )); then
    for epi in "${fmap_epi[@]}"; do
      epi_base="$(strip_nii_ext "$epi")"
      epi_json="${epi_base}.json"
      [[ -f "$JSON_FIXER" && -f "$epi_json" ]] && python3 "$JSON_FIXER" "$epi_json" || true
      ped_fmap="$(read_ped "$epi_json")"; [[ -z "$ped_fmap" ]] && ped_fmap="j"
      dir_tag="AP"; [[ "$ped_fmap" == "j-" ]] && dir_tag="PA"

      # BIDS fmap
      install_bids "$sub" "fmap" "$epi_base"
      fmap_dir="${OUT_RAW}/sub-${sub}/fmap"
      # Handle .nii.gz and .nii files
      if [[ -f "${fmap_dir}/$(basename "$epi_base").nii.gz" ]]; then
        mv -f "${fmap_dir}/$(basename "$epi_base").nii.gz" "${fmap_dir}/sub-${sub}_dir-${dir_tag}_epi.nii.gz"
      elif [[ -f "${fmap_dir}/$(basename "$epi_base").nii" ]]; then
        mv -f "${fmap_dir}/$(basename "$epi_base").nii" "${fmap_dir}/sub-${sub}_dir-${dir_tag}_epi.nii"
      fi
      # Handle JSON
      [[ -f "${fmap_dir}/$(basename "$epi_base").json" ]] && \
        mv -f "${fmap_dir}/$(basename "$epi_base").json" "${fmap_dir}/sub-${sub}_dir-${dir_tag}_epi.json"

      # Add IntendedFor (point to existing DWIs)
      dwi_ap_rel="dwi/sub-${sub}_dir-AP_dwi.nii.gz"
      dwi_pa_rel="dwi/sub-${sub}_dir-PA_dwi.nii.gz"
      if [[ -f "${OUT_RAW}/sub-${sub}/${dwi_ap_rel}" ]] || [[ -f "${OUT_RAW}/sub-${sub}/${dwi_pa_rel}" ]]; then
        intents=()
        [[ -f "${OUT_RAW}/sub-${sub}/${dwi_ap_rel}" ]] && intents+=("$dwi_ap_rel")
        [[ -f "${OUT_RAW}/sub-${sub}/${dwi_pa_rel}" ]] && intents+=("$dwi_pa_rel")
        add_intendedfor "${OUT_RAW}/sub-${sub}/fmap/sub-${sub}_dir-${dir_tag}_epi.json" "${intents[@]}"
      fi

      # Copy same fmap into Preproc (flat)
      install_preproc "$sub" "$epi_base" "sub-${sub}_dir-${dir_tag}_epi"
    done
    log "FMAP (EPI) installed for ${sub}"
  fi

  log "Done: ${sub}"
done

log "All subjects converted."
log "BIDS-like raw: $OUT_RAW"
log "Preproc:       $OUT_PRE (sub-XX/)"
