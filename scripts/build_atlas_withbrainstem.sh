#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Build Combined HCPex Atlas with Brainstem and Cerebellum
# (Glasser + HCPex + Lausanne Brainstem + Lausanne Cerebellum)
# ============================================================================
# This script:
# 1. Loads the existing Combined_HCPex_426_MNI2009c_1mm atlas (labels 1-426)
# 2. Extracts labels 125-128 from Lausanne atlas (brainstem regions)
# 3. Remaps brainstem labels to 427-430
# 4. Extracts labels 62, 124 from Lausanne atlas (cerebellum regions)
# 5. Remaps cerebellum labels to 431-432
# 6. Merges all with the combined atlas
# 7. Saves as Final_Combined_atlas_MNI2009c_1mm.nii.gz
# ============================================================================

# ---- Paths ----
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
ATLAS_DIR="$ROOT/atlas"

# Input files
COMBINED_ATLAS="$ATLAS_DIR/Combined_HCPex_426_MNI2009c_1mm.nii.gz"
LAUSANNE_ATLAS="$ATLAS_DIR/subcort_brainstem/lausanne2018.scale1.sym.corrected.ctx+subc.maxprob.nii"

# Output file
OUTPUT_ATLAS="$ATLAS_DIR/Final_Combined_atlas_MNI2009c_1mm.nii.gz"

# ---- Check inputs ----
if [[ ! -f "$COMBINED_ATLAS" ]]; then
  echo "[ERROR] Combined atlas not found: $COMBINED_ATLAS"
  echo "        Please run build_HCPex_combined_v3_2009c.sh build first."
  exit 1
fi

if [[ ! -f "$LAUSANNE_ATLAS" ]]; then
  echo "[ERROR] Lausanne atlas not found: $LAUSANNE_ATLAS"
  exit 1
fi

# ---- Check if output already exists ----
if [[ -f "$OUTPUT_ATLAS" ]]; then
  echo "[Reuse] Final combined atlas already exists: $OUTPUT_ATLAS"
  echo "        Delete it to rebuild."
  exit 0
fi

# ---- Check if old output exists and warn ----
OLD_OUTPUT="$ATLAS_DIR/Combined_HCPex_brainstem_MNI2009c_1mm.nii.gz"
if [[ -f "$OLD_OUTPUT" ]]; then
  echo "[WARNING] Old output file exists: $OLD_OUTPUT"
  echo "          This script now creates: $OUTPUT_ATLAS"
  echo "          The old file will not be overwritten."
  echo ""
fi

echo "=========================================="
echo "Building Combined Atlas with Brainstem and Cerebellum"
echo "=========================================="
echo ""
echo "Input files:"
echo "  Combined atlas: $COMBINED_ATLAS"
echo "  Lausanne atlas: $LAUSANNE_ATLAS"
echo ""
echo "Output:"
echo "  $OUTPUT_ATLAS"
echo ""

# ---- Verify spaces match ----
echo "[Check] Verifying atlas spaces match..."
COMBINED_SHAPE=$(python3 -c "import nibabel as nib; print(nib.load('$COMBINED_ATLAS').shape[:3])" | tr -d '(),')
LAUSANNE_SHAPE=$(python3 -c "import nibabel as nib; print(nib.load('$LAUSANNE_ATLAS').shape[:3])" | tr -d '(),')

if [[ "$COMBINED_SHAPE" != "$LAUSANNE_SHAPE" ]]; then
  echo "[ERROR] Shape mismatch!"
  echo "        Combined atlas shape: $COMBINED_SHAPE"
  echo "        Lausanne atlas shape: $LAUSANNE_SHAPE"
  echo "        They must be in the same space (MNI152_2009c)."
  exit 1
fi
echo "  ✓ Shapes match: $COMBINED_SHAPE"
echo ""

# ---- Extract and remap brainstem labels ----
echo "[Extract] Extracting labels 125-128 from Lausanne atlas..."
echo "         Remapping: 125→427, 126→428, 127→429, 128→430"

# Create temporary files for each brainstem label
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Extract each label and remap
for old_label in 125 126 127 128; do
  new_label=$((old_label + 302))  # 125+302=427, 126+302=428, etc.
  temp_file="$TEMP_DIR/brainstem_${old_label}_to_${new_label}.nii.gz"
  
  # Extract label: threshold to get only this label, binarize, then multiply by new_label
  # This creates a mask where old_label exists, then assigns new_label to those voxels
  fslmaths "$LAUSANNE_ATLAS" -thr $old_label -uthr $old_label -bin -mul $new_label "$temp_file"
  
  # Verify extraction
  count=$(fslstats "$temp_file" -V | awk '{print $1}')
  echo "  Label $old_label → $new_label: $count voxels"
done
echo ""

# ---- Merge brainstem labels into single file ----
echo "[Merge] Combining brainstem labels (427-430)..."
BRAINSTEM_TEMP="$TEMP_DIR/brainstem_combined.nii.gz"

# Start with label 125→427
fslmaths "$TEMP_DIR/brainstem_125_to_427.nii.gz" "$BRAINSTEM_TEMP"

# Add other labels using max (to combine without overlap)
for old_label in 126 127 128; do
  new_label=$((old_label + 302))
  temp_file="$TEMP_DIR/brainstem_${old_label}_to_${new_label}.nii.gz"
  fslmaths "$BRAINSTEM_TEMP" -max "$temp_file" "$BRAINSTEM_TEMP"
done

# Verify combined brainstem labels
read bmin bmax <<<"$(fslstats "$BRAINSTEM_TEMP" -R)"
echo "  Brainstem labels range: $bmin to $bmax (expect 427-430)"
echo ""

# ---- Extract and remap cerebellum labels ----
echo "[Extract] Extracting labels 62, 124 from Lausanne atlas (cerebellum)..."
echo "         Remapping: 62→431, 124→432"

# Extract each cerebellum label and remap
CEREBELLUM_TEMP="$TEMP_DIR/cerebellum_combined.nii.gz"

# Map 62 → 431 (62 + 369 = 431)
fslmaths "$LAUSANNE_ATLAS" -thr 62 -uthr 62 -bin -mul 431 "$TEMP_DIR/cerebellum_62_to_431.nii.gz"
count_62=$(fslstats "$TEMP_DIR/cerebellum_62_to_431.nii.gz" -V | awk '{print $1}')
echo "  Label 62 → 431: $count_62 voxels"

# Map 124 → 432 (124 + 308 = 432)
fslmaths "$LAUSANNE_ATLAS" -thr 124 -uthr 124 -bin -mul 432 "$TEMP_DIR/cerebellum_124_to_432.nii.gz"
count_124=$(fslstats "$TEMP_DIR/cerebellum_124_to_432.nii.gz" -V | awk '{print $1}')
echo "  Label 124 → 432: $count_124 voxels"
echo ""

# ---- Merge cerebellum labels into single file ----
echo "[Merge] Combining cerebellum labels (431-432)..."
fslmaths "$TEMP_DIR/cerebellum_62_to_431.nii.gz" -max "$TEMP_DIR/cerebellum_124_to_432.nii.gz" "$CEREBELLUM_TEMP"

# Verify combined cerebellum labels
read cmin cmax <<<"$(fslstats "$CEREBELLUM_TEMP" -R)"
echo "  Cerebellum labels range: $cmin to $cmax (expect 431-432)"
echo ""

# ---- Combine brainstem and cerebellum ----
echo "[Merge] Combining brainstem (427-430) and cerebellum (431-432) labels..."
ADDITIONAL_LABELS_TEMP="$TEMP_DIR/additional_labels_combined.nii.gz"
fslmaths "$BRAINSTEM_TEMP" -max "$CEREBELLUM_TEMP" "$ADDITIONAL_LABELS_TEMP"

# Verify combined additional labels
read amin amax <<<"$(fslstats "$ADDITIONAL_LABELS_TEMP" -R)"
echo "  Additional labels range: $amin to $amax (expect 427-432)"
echo ""

# ---- Check for overlaps ----
echo "[Check] Checking for overlaps between Combined atlas and additional regions..."
OVERLAP_TEMP="$TEMP_DIR/overlap_check.nii.gz"
fslmaths "$COMBINED_ATLAS" -bin -mul "$ADDITIONAL_LABELS_TEMP" -bin "$OVERLAP_TEMP"
overlap_count=$(fslstats "$OVERLAP_TEMP" -V | awk '{print $1}')
if [[ "$overlap_count" -gt 0 ]]; then
  echo "  ⚠ WARNING: Found $overlap_count overlapping voxels"
  echo "             These will be assigned the higher label value (427-432)"
else
  echo "  ✓ No overlaps found - clean merge"
fi
echo ""

# ---- Merge with combined atlas ----
echo "[Merge] Merging Combined_HCPex_426 with brainstem (427-430) and cerebellum (431-432) labels..."

# Use max to combine (ensures no overlap, higher labels take precedence)
MERGED_TEMP="$TEMP_DIR/merged_atlas_temp.nii.gz"
fslmaths "$COMBINED_ATLAS" -max "$ADDITIONAL_LABELS_TEMP" "$MERGED_TEMP"

# Convert to uint16 for efficiency and consistency
echo "[Convert] Converting to uint16..."
mrconvert "$MERGED_TEMP" "$OUTPUT_ATLAS" -datatype uint16 -force -quiet

# ---- QC: verify label ranges ----
echo ""
echo "[QC] Quality check:"
read cmin cmax <<<"$(fslstats "$COMBINED_ATLAS" -R)"
read mmin mmax <<<"$(fslstats "$OUTPUT_ATLAS" -R)"

echo "  Combined_HCPex_426 range: $cmin to $cmax (expect 0-426)"
echo "  Output atlas range: $mmin to $mmax (expect 0-432)"

# Count unique labels
COMBINED_COUNT=$(python3 -c "import nibabel as nib; import numpy as np; data = nib.load('$COMBINED_ATLAS').get_fdata(); print(len(np.unique(data[data > 0])))")
OUTPUT_COUNT=$(python3 -c "import nibabel as nib; import numpy as np; data = nib.load('$OUTPUT_ATLAS').get_fdata(); print(len(np.unique(data[data > 0])))")

echo "  Combined_HCPex_426 unique labels: $COMBINED_COUNT (expect 426)"
echo "  Output atlas unique labels: $OUTPUT_COUNT (expect 432)"

# Verify expected labels exist
echo ""
echo "[Verify] Checking for brainstem labels 427-430:"
for label in 427 428 429 430; do
  # Use Python to count exact label matches (more reliable than fslstats for exact values)
  count=$(python3 -c "import nibabel as nib; import numpy as np; data = nib.load('$OUTPUT_ATLAS').get_fdata(); print(int(np.sum(data == $label)))")
  if [[ "$count" -gt 0 ]]; then
    echo "  ✓ Label $label: $count voxels"
  else
    echo "  ✗ Label $label: NOT FOUND"
  fi
done

echo ""
echo "[Verify] Checking for cerebellum labels 431-432:"
for label in 431 432; do
  # Use Python to count exact label matches (more reliable than fslstats for exact values)
  count=$(python3 -c "import nibabel as nib; import numpy as np; data = nib.load('$OUTPUT_ATLAS').get_fdata(); print(int(np.sum(data == $label)))")
  if [[ "$count" -gt 0 ]]; then
    echo "  ✓ Label $label: $count voxels"
  else
    echo "  ✗ Label $label: NOT FOUND"
  fi
done

# Final validation
if awk "BEGIN {exit !($mmax < 427 || $mmax > 432)}"; then
  echo ""
  echo "[WARNING] Output atlas max label ($mmax) is outside expected range (427-432)"
  echo "          This may indicate a problem with the merge."
else
  echo ""
  echo "[OK] Brainstem and cerebellum atlas created successfully: $OUTPUT_ATLAS"
  echo ""
  echo "Summary:"
  echo "  - Labels 1-426: Glasser (1-360) + HCPex (361-426)"
  echo "  - Labels 427-430: Lausanne brainstem (125-128)"
  echo "  - Labels 431-432: Lausanne cerebellum (62, 124)"
  echo "  - Total: 432 labeled regions"
fi

