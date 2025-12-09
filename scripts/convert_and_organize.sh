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
  log "  Processing series in: ${subj_dir}"
  # Get absolute path for subj_dir to avoid path issues
  if ! subj_dir_abs="$(cd "$subj_dir" && pwd 2>/dev/null)"; then
    warn "  Cannot resolve absolute path for: ${subj_dir}"
    subj_dir_abs="$subj_dir"
  fi
  log "  Absolute path: ${subj_dir_abs}"
  
  # Check if directory exists and list contents
  if [[ ! -d "$subj_dir_abs" ]]; then
    warn "  Directory does not exist: ${subj_dir_abs}"
    continue
  fi
  
  dir_contents=$(ls -1 "$subj_dir_abs" 2>/dev/null | head -5 || true)
  log "  Directory contents (first 5): $(echo "$dir_contents" | tr '\n' ' ' || echo 'none')"
  
  series_count=0
  # Use array to handle glob expansion properly
  shopt -s nullglob
  series_array=("$subj_dir_abs"/*)
  shopt -u nullglob
  
  if [[ ${#series_array[@]} -eq 0 ]] || [[ ! -e "${series_array[0]}" ]]; then
    warn "  No series directories found"
  else
    log "  Found ${#series_array[@]} item(s) to check"
    
    # First, check for NIfTI files directly in subject root (e.g., already-converted T1 files)
    log "  Checking for NIfTI files in subject root..."
    shopt -s nullglob
    for root_file in "$subj_dir_abs"/*.{nii,nii.gz,json,bval,bvec} "$subj_dir_abs"/*.{NII,NII.GZ,JSON,BVAL,BVEC}; do
      if [[ -f "$root_file" ]]; then
        log "    Found root-level file: $(basename "$root_file")"
        # Check if file is readable before attempting to copy
        if [[ -r "$root_file" ]]; then
          if cp -f "$root_file" "$sub_tmp/" 2>&1; then
            log "      Copied: $(basename "$root_file")"
          else
            cp_exit_code=$?
            warn "      Failed to copy: $(basename "$root_file") (exit code: $cp_exit_code)"
            # Try to get more details about why copy failed
            if [[ ! -r "$root_file" ]]; then
              warn "      File is not readable (permissions may have changed)"
            elif [[ ! -w "$sub_tmp" ]]; then
              warn "      Destination directory is not writable"
            fi
          fi
        else
          warn "      Permission denied (not readable): $(basename "$root_file") - file exists but cannot be read"
          warn "      File permissions: $(stat -c "%a %U:%G" "$root_file" 2>/dev/null || echo "unknown")"
          warn "      Current user groups: $(groups 2>/dev/null | head -c 100 || echo "unknown")"
          warn "      Try: chmod +r \"$root_file\" to fix permissions"
        fi
      fi
    done
    shopt -u nullglob
    
    # Now process series directories (handle zip files)
    # First pass: extract zip files and collect all directories to process
    directories_to_process=()
    # First pass: identify zip files and check if corresponding directories already exist
    zip_dirs_to_skip=()
    for series in "${series_array[@]}"; do
      if [[ "$series" == *.zip ]] || [[ "$series" == *.ZIP ]]; then
        # Check if corresponding directory already exists (without timestamp)
        extract_base="${series%.zip}"
        extract_base="${extract_base%.ZIP}"
        extract_parent="$(dirname "$series")"
        base_no_timestamp=$(basename "$extract_base" | sed 's/-[0-9]\{8\}T[0-9]\{6\}Z-[0-9]-[0-9]\{3\}$//' 2>/dev/null || echo "") || true
        if [[ -n "$base_no_timestamp" ]]; then
          if [[ "$extract_parent" == "." ]]; then
            candidate_dir="$base_no_timestamp"
          else
            candidate_dir="$extract_parent/$base_no_timestamp"
          fi
          # Get absolute path for comparison
          if [[ -d "$candidate_dir" ]]; then
            candidate_abs="$(cd "$candidate_dir" && pwd 2>/dev/null || echo "$candidate_dir")"
            zip_dirs_to_skip+=("$candidate_abs")
          fi
        fi
      fi
    done
    
    # Second pass: add directories and extract zips, skipping directories that correspond to zip files
    for series in "${series_array[@]}"; do
      if [[ -d "$series" ]]; then
        # Check if this directory corresponds to a zip file we'll extract
        series_abs="$(cd "$series" && pwd 2>/dev/null || echo "$series")"
        skip_dir=false
        for zip_dir_abs in "${zip_dirs_to_skip[@]}"; do
          if [[ "$series_abs" == "$zip_dir_abs" ]]; then
            skip_dir=true
            log "    Skipping directory (corresponding zip file will be processed): $(basename "$series")"
            break
          fi
        done
        if [[ "$skip_dir" == "false" ]]; then
          directories_to_process+=("$series")
        fi
      elif [[ "$series" == *.zip ]] || [[ "$series" == *.ZIP ]]; then
        # Check if it's a zip file - extract it first
        log "    Found zip file: $(basename "$series") - extracting..."
        extract_base="${series%.zip}"
        extract_base="${extract_base%.ZIP}"
        extract_parent="$(dirname "$series")"
        if command -v unzip >/dev/null 2>&1; then
          # First, try to determine what directory name is inside the zip
          zip_internal_dir=""
          if unzip -l "$series" 2>/dev/null | head -5 | grep -q "/" 2>/dev/null || true; then
            # Get the first directory path from zip listing
            zip_internal_dir=$(unzip -l "$series" 2>/dev/null | grep -E "^[[:space:]]*[0-9]+.*/" | head -1 | sed 's/.*[0-9][0-9]:[0-9][0-9][[:space:]]*//' | sed 's/\/.*$//' | head -1 || echo "") || true
          fi
          
          # Extract the zip file
          unzip_output=$(unzip -q -o "$series" -d "$extract_parent" 2>&1) || true
          unzip_exit_code=$?
          
          if [[ $unzip_exit_code -eq 0 ]] || [[ -z "$unzip_output" ]]; then
            log "    Extracted zip file"
            # Find what directory was actually created (may differ from zip name)
            found_extracted_dir=""
            
            # Strategy 1: If we found the internal directory name, use it
            if [[ -n "$zip_internal_dir" ]]; then
              # Handle both absolute and relative paths
              if [[ "$extract_parent" == "." ]]; then
                candidate_dir="$zip_internal_dir"
              else
                candidate_dir="$extract_parent/$zip_internal_dir"
              fi
              if [[ -d "$candidate_dir" ]]; then
                found_extracted_dir="$candidate_dir"
                log "    Found directory from zip contents: $zip_internal_dir"
              fi
            fi
            
            # Strategy 2: Try exact match (with timestamp) if Strategy 1 didn't work
            if [[ -z "$found_extracted_dir" ]] && [[ -d "$extract_base" ]]; then
              found_extracted_dir="$extract_base"
            fi
            
            # Strategy 3: Try without timestamp suffix if still not found
            if [[ -z "$found_extracted_dir" ]]; then
              base_no_timestamp=$(basename "$extract_base" | sed 's/-[0-9]\{8\}T[0-9]\{6\}Z-[0-9]-[0-9]\{3\}$//' 2>/dev/null || echo "") || true
              if [[ -n "$base_no_timestamp" ]]; then
                if [[ "$extract_parent" == "." ]]; then
                  candidate_dir="$base_no_timestamp"
                else
                  candidate_dir="$extract_parent/$base_no_timestamp"
                fi
                if [[ -d "$candidate_dir" ]]; then
                  found_extracted_dir="$candidate_dir"
                fi
              fi
            fi
            
            # Strategy 4: Find the most recently created/modified directory in parent if still not found
            if [[ -z "$found_extracted_dir" ]]; then
              found_extracted_dir=$(find "$extract_parent" -maxdepth 1 -type d -newer "$series" ! -path "$extract_parent" 2>/dev/null | head -1 || echo "") || true
              # If that didn't work, try finding directories that were modified around the same time
              if [[ -z "$found_extracted_dir" ]]; then
                zip_mtime=$(stat -c %Y "$series" 2>/dev/null || echo "0") || true
                if [[ -n "$zip_mtime" ]] && [[ "$zip_mtime" != "0" ]]; then
                  # Use a safer approach: find all directories and check their mtime
                  shopt -s nullglob || true
                  for candidate_dir in "$extract_parent"/*/; do
                    if [[ -d "$candidate_dir" ]] && [[ "$candidate_dir" != "$extract_parent/" ]]; then
                      candidate_mtime=$(stat -c %Y "$candidate_dir" 2>/dev/null || echo "0") || true
                      # Safely compare mtimes - ensure both are valid numbers
                      if [[ -n "$candidate_mtime" ]] && [[ "$candidate_mtime" != "0" ]] && \
                         [[ -n "$zip_mtime" ]] && [[ "$zip_mtime" != "0" ]] && \
                         [[ "$candidate_mtime" =~ ^[0-9]+$ ]] && [[ "$zip_mtime" =~ ^[0-9]+$ ]] && \
                         (( candidate_mtime >= zip_mtime )); then
                        found_extracted_dir="${candidate_dir%/}"
                        break
                      fi
                    fi
                  done
                  shopt -u nullglob || true
                fi
              fi
            fi
            
            # Add extracted directory to processing list
            if [[ -n "$found_extracted_dir" ]] && [[ -d "$found_extracted_dir" ]]; then
              directories_to_process+=("$found_extracted_dir") || true
              log "    Found extracted directory: $(basename "$found_extracted_dir")"
            else
              warn "    Could not locate extracted directory for: $(basename "$series")"
              warn "    Zip may have extracted to an unexpected location or directory already exists"
              # If directory already exists (not extracted), check if we should process it anyway
              base_no_timestamp=$(basename "$extract_base" | sed 's/-[0-9]\{8\}T[0-9]\{6\}Z-[0-9]-[0-9]\{3\}$//' 2>/dev/null || echo "") || true
              if [[ -n "$base_no_timestamp" ]] && [[ -d "$extract_parent/$base_no_timestamp" ]]; then
                log "    Using existing directory: $base_no_timestamp"
                directories_to_process+=("$extract_parent/$base_no_timestamp") || true
              fi
            fi
          else
            warn "    Failed to extract zip file: $(basename "$series")"
            warn "    unzip exit code: $unzip_exit_code"
            if [[ -n "$unzip_output" ]]; then
              warn "    unzip output: ${unzip_output:0:200}"
            fi
            # If extraction failed but directory already exists, use it
            base_no_timestamp=$(basename "$extract_base" | sed 's/-[0-9]\{8\}T[0-9]\{6\}Z-[0-9]-[0-9]\{3\}$//' 2>/dev/null || echo "") || true
            if [[ -n "$base_no_timestamp" ]] && [[ -d "$extract_parent/$base_no_timestamp" ]]; then
              log "    Using existing directory (extraction may have failed because it already exists): $base_no_timestamp"
              directories_to_process+=("$extract_parent/$base_no_timestamp") || true
            fi
          fi
        else
          warn "    unzip command not available - cannot extract: $(basename "$series")"
        fi
      else
        log "    Skipping non-directory: $(basename "$series")"
      fi
    done
    
    # Second pass: process all directories (original + extracted)
    for series in "${directories_to_process[@]}"; do
      ((series_count++)) || true  # || true to prevent exit on arithmetic
      series_name="$(basename "$series")"
      log "  [${series_count}] Checking series: ${series_name}"
      
      # Get absolute path to handle spaces and special characters
      if ! series_abs="$(cd "$series" && pwd 2>/dev/null)"; then
        warn "    Cannot access directory: ${series_name}"
        continue
      fi
      log "    Absolute path: ${series_abs}"
      
      # Check if directory contains NIfTI files (already converted)
      has_nifti=false
      nifti_files=$(find "$series_abs" -maxdepth 1 -type f \( -name "*.nii" -o -name "*.nii.gz" \) 2>/dev/null | wc -l)
      if [[ "$nifti_files" -gt 0 ]]; then
        has_nifti=true
        log "    Found ${nifti_files} NIfTI file(s)"
      fi
      
      # Check if directory contains DICOM files
      has_dicom=false
      # Check for DICOM with extensions
      dicom_ext_files=$(find "$series_abs" -maxdepth 2 -type f \( -name "*.dcm" -o -name "DICOMDIR" -o -name "*.DCM" \) 2>/dev/null | wc -l)
      if [[ "$dicom_ext_files" -gt 0 ]]; then
        has_dicom=true
        log "    Found ${dicom_ext_files} DICOM file(s) with extensions"
      else
        # Check for extensionless files that might be DICOM
        file_count=$(find "$series_abs" -maxdepth 1 -type f ! -name ".*" ! -name "*.nii" ! -name "*.nii.gz" ! -name "*.json" ! -name "*.bval" ! -name "*.bvec" ! -name "*.bvals" ! -name "*.bvecs" ! -name "*.txt" ! -name "*.log" 2>/dev/null | wc -l || echo "0")
        file_count=$((file_count + 0))  # Remove whitespace and ensure it's a number
        log "    Found ${file_count} extensionless file(s)"
        if [[ "$file_count" -gt 0 ]]; then
          # Sample a file to check if it's DICOM
          sample_file=$(find "$series_abs" -maxdepth 1 -type f ! -name ".*" ! -name "*.nii" ! -name "*.nii.gz" ! -name "*.json" ! -name "*.bval" ! -name "*.bvec" ! -name "*.bvals" ! -name "*.bvecs" 2>/dev/null | head -1 || echo "")
          log "    Sample file result: ${sample_file:0:50}..."
          if [[ -n "$sample_file" ]] && [[ -f "$sample_file" ]]; then
            log "    Sampling file: $(basename "$sample_file")"
            if command -v file >/dev/null 2>&1; then
              if file_type=$(file "$sample_file" 2>/dev/null); then
                log "    File type: ${file_type}"
                if echo "$file_type" | grep -qi "DICOM" 2>/dev/null; then
                  has_dicom=true
                  log "    Detected as DICOM"
                fi
              else
                # If file command fails, assume it's DICOM (common for extensionless files)
                has_dicom=true
                log "    Assuming DICOM (file command failed, likely DICOM)"
              fi
            else
              # If 'file' command not available, assume extensionless files in series dirs are DICOM
              has_dicom=true
              log "    Assuming DICOM (file command not available)"
            fi
          else
            log "    No sample file found (unexpected)"
          fi
        fi
      fi
      
      if [[ "$has_nifti" == "true" ]]; then
        # Copy already-converted NIfTI files
        log "    Copying NIfTI files: ${series_name}"
        copied=0
        shopt -s nullglob
        for nii_file in "$series_abs"/*.{nii,nii.gz,json,bval,bvec} "$series_abs"/*.{NII,NII.GZ,JSON,BVAL,BVEC}; do
          if [[ -f "$nii_file" ]]; then
            cp -f "$nii_file" "$sub_tmp/" && ((copied++)) || true
          fi
        done
        shopt -u nullglob
        log "    Copied ${copied} file(s) to temp directory"
      elif [[ "$has_dicom" == "true" ]]; then
        # Convert DICOM to NIfTI - use absolute path and cd into directory for dcm2niix
        log "    Running dcm2niix: ${series_name}"
        log "    Working directory: ${series_abs}"
        log "    Output directory: ${sub_tmp}"
        dcm2niix_output=$(cd "$series_abs" && "$DCM2NIIX" -z y -f "%p_%s" -o "$sub_tmp" . 2>&1)
        dcm2niix_exit=$?
        if [[ $dcm2niix_exit -eq 0 ]]; then
          log "    dcm2niix succeeded"
          echo "$dcm2niix_output" | grep -E "(Convert|Found)" | while read line; do log "      $line"; done || true
        else
          warn "    dcm2niix failed (exit code: $dcm2niix_exit) for ${series_name}"
          echo "$dcm2niix_output" | head -5 | while read line; do warn "      $line"; done || true
        fi
      else
        # Check if directory has bval/bvec/json files but no NIfTI (e.g., sub-FD, sub-TL)
        # These might be orphaned files that need to be matched with NIfTI from elsewhere
        # Note: Some files have .bvals/.bvecs (plural) instead of .bval/.bvec (singular)
        log "    No NIfTI or DICOM found, checking for auxiliary files (bval/bvec/json)..."
        has_aux_files=false
        aux_count=0
        # Check for auxiliary files with explicit find to avoid glob issues
        while IFS= read -r aux_file; do
          if [[ -f "$aux_file" ]]; then
            has_aux_files=true
            aux_count=$((aux_count + 1))
            log "      Found auxiliary file: $(basename "$aux_file")"
          fi
        done < <(find "$series_abs" -maxdepth 1 -type f \( -name "*.json" -o -name "*.bval" -o -name "*.bvec" -o -name "*.bvals" -o -name "*.bvecs" -o -name "*.JSON" -o -name "*.BVAL" -o -name "*.BVEC" -o -name "*.BVALS" -o -name "*.BVECS" \) 2>/dev/null || true)
        
        if [[ "$has_aux_files" == "true" ]]; then
          log "    Found ${aux_count} auxiliary file(s) (json/bval/bvec) but no NIfTI or DICOM - copying for later matching"
          copied=0
          shopt -s nullglob
          for aux_file in "$series_abs"/*.{json,bval,bvec,bvals,bvecs} "$series_abs"/*.{JSON,BVAL,BVEC,BVALS,BVECS}; do
            if [[ -f "$aux_file" ]]; then
              if cp -f "$aux_file" "$sub_tmp/" 2>/dev/null; then
                copied=$((copied + 1))  # Use $((...)) instead of ((...)) to avoid set -e issues
                log "      Copied: $(basename "$aux_file")"
              else
                warn "      Failed to copy: $(basename "$aux_file")"
              fi
            fi
          done
          shopt -u nullglob
          log "    Copied ${copied} auxiliary file(s) to temp directory"
        else
          warn "    No DICOM, NIfTI, or auxiliary files found in: ${series_name}"
        fi
      fi
    done  # End of for series loop
  fi  # End of if series_array check
  log "  Processed ${series_count} series"

  # Identify series (handle both .nii and .nii.gz, flexible patterns)
  log "  Identifying converted files in: ${sub_tmp}"
  temp_file_count=$(find "$sub_tmp" -maxdepth 1 -type f 2>/dev/null | wc -l)
  log "    Found ${temp_file_count} file(s) in temp directory"
  shopt -s nullglob nocaseglob
  # DWI patterns: match diff/MDDW/DTI files (with or without "64" in name, handle "68 directions" too)
  dwi_main=( "$sub_tmp"/*diff*64*.nii.gz "$sub_tmp"/*MDDW*64*.nii.gz \
             "$sub_tmp"/*diff*64*.nii "$sub_tmp"/*MDDW*64*.nii \
             "$sub_tmp"/*DTI*.nii.gz "$sub_tmp"/*DTI*.nii \
             "$sub_tmp"/*diff*.nii.gz "$sub_tmp"/*MDDW*.nii.gz \
             "$sub_tmp"/*diff*.nii "$sub_tmp"/*MDDW*.nii \
             "$sub_tmp"/*ep2d*diff*.nii.gz "$sub_tmp"/*ep2d*diff*.nii )
  
  # If no DWI files found by pattern, check for bval/bvec files that might indicate DWI
  if (( ${#dwi_main[@]} == 0 )); then
    log "    No DWI files found by filename pattern, checking for bval/bvec files..."
    # Check if we have bval/bvec files but no matching NIfTI
    for bval_file in "$sub_tmp"/*.bval "$sub_tmp"/*.bvals; do
      [[ -f "$bval_file" ]] || continue
      bval_base="${bval_file%.bval}"
      bval_base="${bval_base%.bvals}"
      # Check if corresponding NIfTI exists
      found_matching_nii=false
      for nii_candidate in "${bval_base}".{nii,nii.gz}; do
        if [[ -f "$nii_candidate" ]] && [[ ! "$nii_candidate" =~ [Bb]0 ]] && [[ ! "$nii_candidate" =~ _b0_ ]]; then
          dwi_main+=("$nii_candidate")
          log "    Found DWI via bval file: $(basename "$nii_candidate")"
          found_matching_nii=true
          break
        fi
      done
      # If no matching NIfTI found, check all NIfTI files in temp directory
      # BUT exclude T1 files (they should not be matched to DWI bval/bvec files)
      if [[ "$found_matching_nii" == "false" ]]; then
        # Count how many NIfTI files don't have bval/bvec AND are not T1 files
        unmatched_nii_count=0
        unmatched_nii_file=""
        for nii_candidate in "$sub_tmp"/*.{nii,nii.gz}; do
          if [[ -f "$nii_candidate" ]] && [[ ! "$nii_candidate" =~ [Bb]0 ]] && [[ ! "$nii_candidate" =~ _b0_ ]]; then
            # Skip T1 files - they should not be matched to DWI bval/bvec
            if [[ "$(basename "$nii_candidate")" =~ [Tt]1 ]] || [[ "$(basename "$nii_candidate")" =~ t1_mprage ]] || [[ "$(basename "$nii_candidate")" =~ T1w ]]; then
              continue
            fi
            nii_base="${nii_candidate%.nii.gz}"
            nii_base="${nii_base%.nii}"
            # If this NIfTI has no bval/bvec, it might match our bval file
            if [[ ! -f "${nii_base}.bval" ]] && [[ ! -f "${nii_base}.bvec" ]] && [[ ! -f "${nii_base}.bvals" ]] && [[ ! -f "${nii_base}.bvecs" ]]; then
              unmatched_nii_count=$((unmatched_nii_count + 1))
              unmatched_nii_file="$nii_candidate"
            fi
          fi
        done
        # If there's exactly one unmatched NIfTI (and it's not a T1) and we have bval/bvec files, match them
        if [[ "$unmatched_nii_count" -eq 1 ]] && [[ -n "$unmatched_nii_file" ]]; then
          dwi_main+=("$unmatched_nii_file")
          log "    Matched single unmatched NIfTI to bval/bvec files: $(basename "$unmatched_nii_file")"
          found_matching_nii=true
        fi
      fi
      if [[ "$found_matching_nii" == "true" ]]; then
        break
      fi
    done
  fi
  
  # If still no DWI files found, check JSON files to identify DWI by metadata
  if (( ${#dwi_main[@]} == 0 )); then
    log "    No DWI files found by filename pattern, checking JSON metadata..."
    for json_file in "$sub_tmp"/*.json; do
      [[ -f "$json_file" ]] || continue
      json_base="${json_file%.json}"
      
      # Check if JSON indicates DWI by checking SeriesDescription/ProtocolName or DiffusionDirectionality
      is_dwi=false
      if command -v jq >/dev/null 2>&1; then
        # Check for DiffusionDirectionality field (present in DWI JSONs)
        if jq -e '.DiffusionDirectionality // empty | length > 0' "$json_file" >/dev/null 2>&1; then
          is_dwi=true
          log "    JSON has DiffusionDirectionality: $(basename "$json_file")"
        # Check SeriesDescription/ProtocolName for DWI keywords
        elif series_desc=$(jq -r '(.SeriesDescription // "") + " " + (.ProtocolName // "")' "$json_file" 2>/dev/null); then
          if echo "$series_desc" | grep -qiE "(diff|MDDW|DTI|ep2d.*diff)"; then
            is_dwi=true
            log "    JSON SeriesDescription/ProtocolName indicates DWI: $(basename "$json_file")"
          fi
        fi
      else
        # Fallback: check JSON filename or content with grep
        if [[ "$(basename "$json_file")" =~ [Dd][Tt][Ii] ]] || \
           [[ "$(basename "$json_file")" =~ [Dd][Ii][Ff][Ff] ]] || \
           grep -qiE "(diff|MDDW|DTI|ep2d.*diff)" "$json_file" 2>/dev/null; then
          is_dwi=true
          log "    JSON filename/content suggests DWI: $(basename "$json_file")"
        fi
      fi
      
      if [[ "$is_dwi" == "true" ]]; then
        # Find corresponding NIfTI file (try exact match first)
        found_nii=false
        for nii_candidate in "${json_base}".{nii,nii.gz}; do
          if [[ -f "$nii_candidate" ]]; then
            dwi_main+=("$nii_candidate")
            log "    Found matching NIfTI: $(basename "$nii_candidate")"
            found_nii=true
            break
          fi
        done
        
        # If no exact match, check all NIfTI files and match by checking if they have bval/bvec or if JSON name suggests match
        if [[ "$found_nii" == "false" ]]; then
          # First, check if there are bval/bvec files that might match
          for bval_file in "$sub_tmp"/*.bval "$sub_tmp"/*.bvals; do
            [[ -f "$bval_file" ]] || continue
            bval_base="${bval_file%.bval}"
            bval_base="${bval_base%.bvals}"
            # Check if corresponding NIfTI exists
            for nii_candidate in "${bval_base}".{nii,nii.gz}; do
              if [[ -f "$nii_candidate" ]] && [[ ! "$nii_candidate" =~ [Bb]0 ]] && [[ ! "$nii_candidate" =~ _b0_ ]]; then
                # If this NIfTI doesn't have a JSON or its JSON matches our DWI JSON, it's a match
                nii_base="${nii_candidate%.nii.gz}"
                nii_base="${nii_base%.nii}"
                if [[ ! -f "${nii_base}.json" ]] || [[ "$(readlink -f "${nii_base}.json" 2>/dev/null || echo "${nii_base}.json")" == "$(readlink -f "$json_file" 2>/dev/null || echo "$json_file")" ]]; then
                  dwi_main+=("$nii_candidate")
                  log "    Found DWI via bval/bvec: $(basename "$nii_candidate")"
                  found_nii=true
                  break 2
                fi
              fi
            done
          done
          
          # If still not found, check for unmatched NIfTI files (no bval/bvec) when JSON has DTI/diff in name
          # This handles cases where bval/bvec/json are in a separate directory from the NIfTI
          # BUT exclude T1 files - they should not be matched to DWI
          if [[ "$found_nii" == "false" ]] && [[ "$(basename "$json_file")" =~ [Dd][Tt][Ii] ]]; then
            unmatched_nii=()
            for nii_candidate in "$sub_tmp"/*.{nii,nii.gz}; do
              if [[ -f "$nii_candidate" ]] && [[ ! "$nii_candidate" =~ [Bb]0 ]] && [[ ! "$nii_candidate" =~ _b0_ ]]; then
                # Skip T1 files - they should not be matched to DWI
                if [[ "$(basename "$nii_candidate")" =~ [Tt]1 ]] || [[ "$(basename "$nii_candidate")" =~ t1_mprage ]] || [[ "$(basename "$nii_candidate")" =~ T1w ]]; then
                  continue
                fi
                nii_base="${nii_candidate%.nii.gz}"
                nii_base="${nii_base%.nii}"
                # If this NIfTI has no bval/bvec files, it might match our DWI JSON/bval/bvec
                if [[ ! -f "${nii_base}.bval" ]] && [[ ! -f "${nii_base}.bvec" ]] && [[ ! -f "${nii_base}.bvals" ]] && [[ ! -f "${nii_base}.bvecs" ]]; then
                  unmatched_nii+=("$nii_candidate")
                fi
              fi
            done
            # If we have exactly one unmatched NIfTI and one DWI JSON, match them
            if (( ${#unmatched_nii[@]} == 1 )); then
              dwi_main+=("${unmatched_nii[0]}")
              log "    Matched single unmatched NIfTI to DTI JSON: $(basename "${unmatched_nii[0]}")"
              found_nii=true
            elif (( ${#unmatched_nii[@]} > 1 )); then
              # Multiple unmatched - try to match by checking if JSON filename suggests a match
              # For example, if JSON is "Subject_DTI.json" and we have "data.nii", match them
              log "    Found ${#unmatched_nii[@]} unmatched NIfTI files, attempting to match..."
              # Try generic names first (data.nii, image.nii, etc.)
              for generic_name in data image; do
                for nii_candidate in "${unmatched_nii[@]}"; do
                  if [[ "$(basename "$nii_candidate")" == "${generic_name}.nii" ]] || [[ "$(basename "$nii_candidate")" == "${generic_name}.nii.gz" ]]; then
                    dwi_main+=("$nii_candidate")
                    log "    Matched generic-named NIfTI to DTI JSON: $(basename "$nii_candidate")"
                    found_nii=true
                    break 2
                  fi
                done
              done
            fi
          fi
        fi
      fi
    done
  fi
  
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
  log "  Searching for DWI files: found ${#dwi_main[@]} candidate(s)"
  if (( ${#dwi_main[@]} )); then
    log "    DWI file: $(basename "${dwi_main[0]}")"
    dwi_main_nii="${dwi_main[0]}"; dwi_base="$(strip_nii_ext "$dwi_main_nii")"
    dwi_json="${dwi_base}.json"
    
    # If JSON file doesn't exist with same base name, try to find matching JSON file
    # This handles cases where NIfTI is named "data.nii" but JSON is named "SUBJECT_DTI.json"
    if [[ ! -f "$dwi_json" ]]; then
      # Look for DTI/diff JSON files in temp directory that might match
      for candidate_json in "$sub_tmp"/*DTI*.json "$sub_tmp"/*diff*.json "$sub_tmp"/*MDDW*.json; do
        if [[ -f "$candidate_json" ]]; then
          # Check if this JSON file is actually for DWI (has DiffusionDirectionality or DWI keywords)
          is_dwi_json=false
          if command -v jq >/dev/null 2>&1; then
            if jq -e '.DiffusionDirectionality // empty | length > 0' "$candidate_json" >/dev/null 2>&1 || \
               jq -r '(.SeriesDescription // "") + " " + (.ProtocolName // "")' "$candidate_json" 2>/dev/null | grep -qiE "(diff|MDDW|DTI|ep2d.*diff)"; then
              is_dwi_json=true
            fi
          else
            # Fallback: check filename or content
            if [[ "$(basename "$candidate_json")" =~ [Dd][Tt][Ii] ]] || \
               [[ "$(basename "$candidate_json")" =~ [Dd][Ii][Ff][Ff] ]] || \
               grep -qiE "(diff|MDDW|DTI|ep2d.*diff|DiffusionDirectionality)" "$candidate_json" 2>/dev/null; then
              is_dwi_json=true
            fi
          fi
          if [[ "$is_dwi_json" == "true" ]]; then
            dwi_json="$candidate_json"
            log "    Found matching DWI JSON: $(basename "$dwi_json")"
            break
          fi
        fi
      done
    fi
    
    [[ -f "$JSON_FIXER" && -f "$dwi_json" ]] && python3 "$JSON_FIXER" "$dwi_json" || true

    # If bval/bvec files don't exist with same base name as NIfTI, look for them with different names
    # This handles cases where dcm2niix creates "data.nii" but bval/bvec have descriptive names
    if [[ ! -f "${dwi_base}.bval" ]] || [[ ! -f "${dwi_base}.bvec" ]]; then
      log "    bval/bvec files not found with base name '$(basename "$dwi_base")', searching for matching files..."
      # Look for bval/bvec files that might match this DWI (check all bval files in temp directory)
      for candidate_bval in "$sub_tmp"/*.bval "$sub_tmp"/*.bvals; do
        [[ -f "$candidate_bval" ]] || continue
        candidate_base="${candidate_bval%.bval}"
        candidate_base="${candidate_base%.bvals}"
        candidate_bvec="${candidate_base}.bvec"
        [[ ! -f "$candidate_bvec" ]] && candidate_bvec="${candidate_base}.bvecs"
        [[ ! -f "$candidate_bvec" ]] && continue
        
        # Check if this bval/bvec pair matches our DWI by checking:
        # 1. If there's a matching NIfTI with same base name (already handled above)
        # 2. If the bval/bvec files are for DWI (have multiple non-zero b-values)
        # 3. If there's only one DWI NIfTI without bval/bvec, match them
        if [[ -f "${candidate_base}.nii" ]] || [[ -f "${candidate_base}.nii.gz" ]]; then
          # This bval/bvec already has a matching NIfTI, skip
          continue
        fi
        
        # Check if this bval file has DWI-like content (multiple b-values, not all zeros)
        if command -v awk >/dev/null 2>&1; then
          non_zero_count=$(awk '{for(i=1;i<=NF;i++) if($i>0) count++} END {print count+0}' "$candidate_bval" 2>/dev/null || echo "0")
          if [[ "$non_zero_count" -gt 1 ]]; then
            # This looks like a DWI bval file, and we have a DWI NIfTI without bval/bvec
            # Copy/link the bval/bvec files to match our NIfTI base name
            cp -f "$candidate_bval" "${dwi_base}.bval" 2>/dev/null || true
            cp -f "$candidate_bvec" "${dwi_base}.bvec" 2>/dev/null || true
            log "    Matched bval/bvec files '$(basename "$candidate_base")' to DWI NIfTI '$(basename "$dwi_base")'"
            break
          fi
        fi
      done
    fi

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
    # Copy JSON file if it exists (even if name doesn't match NIfTI base name)
    if [[ -f "$dwi_json" ]] && [[ "$dwi_json" != "${dwi_base}.json" ]]; then
      # Copy the JSON file to match the NIfTI base name for BIDS compatibility
      cp -f "$dwi_json" "${dwi_base}.json" 2>/dev/null || true
      log "    Copied JSON file to match NIfTI base name: $(basename "${dwi_base}.json")"
    fi
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

    # Preproc dwi - use the BIDS directory path after files have been moved
    pre_tag="$(pe_to_preproc_tag "$ped")"
    dwi_bids_base="${OUT_RAW}/sub-${sub}/dwi/sub-${sub}_dir-${dir_tag}_dwi"
    install_preproc "$sub" "$dwi_bids_base" "sub-${sub}_${pre_tag}_dwi"
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
      # Handle JSON, bval, and bvec (b0 files may have bval/bvec if dcm2niix created them)
      for ext in json bval bvec; do
        [[ -f "$(basename "$b0_base").${ext}" ]] && mv -f "$(basename "$b0_base").${ext}" "sub-${sub}_dir-${dir_tag_b0}_b0.${ext}"
      done
    popd >/dev/null

    # Preproc b0 - use the BIDS directory path after files have been moved
    b0_bids_base="${OUT_RAW}/sub-${sub}/dwi/sub-${sub}_dir-${dir_tag_b0}_b0"
    install_preproc "$sub" "$b0_bids_base" "sub-${sub}_$(pe_to_preproc_tag "$ped_b0_final")_b0"
  fi

  # --- T1 (also into Preproc) ---
  log "  Searching for T1 files: found ${#anat_t1[@]} candidate(s)"
  if (( ${#anat_t1[@]} )); then
    log "    T1 file: $(basename "${anat_t1[0]}")"
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
