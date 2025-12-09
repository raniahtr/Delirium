#!/bin/bash
# Batch script to run voxelwise SNR QC for all subjects in Preproc_current

set -u 

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREPROC_DIR="${1:-/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_current}"
SNR_SCRIPT="${SCRIPT_DIR}/snr_voxelwise_qc.py"

echo "=========================================="
echo "Batch SNR Voxelwise QC"
echo "=========================================="
echo "Preprocessing directory: $PREPROC_DIR"
echo "SNR script: $SNR_SCRIPT"
echo "=========================================="
echo ""

# Check if script exists
if [[ ! -f "$SNR_SCRIPT" ]]; then
    echo "ERROR: SNR script not found: $SNR_SCRIPT"
    exit 1
fi

# Count subjects
SUBJECTS=("$PREPROC_DIR"/sub-*)
N_SUBJECTS=${#SUBJECTS[@]}
echo "Found $N_SUBJECTS subjects"
echo ""

# Process each subject
SUCCESS=0
FAILED=0
SKIPPED=0

for subj_dir in "${SUBJECTS[@]}"; do
    SUBJ=$(basename "$subj_dir")
    
    # Check if required files exist
    DWI_RAW="$subj_dir/${SUBJ}_dwi.mif"
    B0_PREPROC="$subj_dir/mean_b0.mif"
    NOISE="$subj_dir/noise.mif"
    OUT_DIR="$subj_dir/qc_snr"
    
    if [[ ! -f "$DWI_RAW" ]]; then
        echo "[SKIP] $SUBJ - Raw DWI not found: $DWI_RAW"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    if [[ ! -f "$B0_PREPROC" ]]; then
        echo "[SKIP] $SUBJ - Preprocessed b0 not found: $B0_PREPROC"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    if [[ ! -f "$NOISE" ]]; then
        echo "[SKIP] $SUBJ - Noise map not found: $NOISE"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Check if already processed (optional - comment out if you want to reprocess)
    if [[ -f "$OUT_DIR/${SUBJ}_snr_comparison_stats.json" ]]; then
        echo "[SKIP] $SUBJ - Already processed (stats file exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    echo "----------------------------------------"
    echo "Processing: $SUBJ"
    echo "----------------------------------------"
    
    # Run SNR QC
    if python3 "$SNR_SCRIPT" \
        --sub_dir "$subj_dir" \
        --sub_id "$SUBJ" \
        --out_dir "$OUT_DIR" 2>&1; then
        echo "[OK] $SUBJ - Completed successfully"
        SUCCESS=$((SUCCESS + 1))
    else
        EXIT_CODE=$?
        echo "[FAIL] $SUBJ - Processing failed (exit code: $EXIT_CODE)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "BATCH PROCESSING COMPLETE"
echo "=========================================="
echo "Total subjects:  $N_SUBJECTS"
echo "Successful:      $SUCCESS"
echo "Failed:          $FAILED"
echo "Skipped:         $SKIPPED"
echo "=========================================="

# Exit with error if any failed
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi

exit 0

