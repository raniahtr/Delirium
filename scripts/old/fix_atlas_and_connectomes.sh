#!/usr/bin/env bash
set -euo pipefail
ROOT="/media/RCPNAS/Data/Delirium/Delirium_Rania"
ATLAS_MNI="$ROOT/atlas/HCPMMP1_labels_3D.nii.gz"
MNI_T1=${FSLDIR:-/usr/local/fsl}/data/standard/MNI152_T1_1mm.nii.gz

for sdir in sub-*; do
  [[ -d "$sdir" ]] || continue
  cd "$sdir"
  SUBJ="$sdir"

  # Skip if required inputs missing
  [[ -f mean_b0.mif && -f diff2struct_mrtrix.txt ]] || { cd ..; continue; }

  # Ensure T1.nii.gz
  [[ -f T1.nii.gz ]] || { [[ -f T1.mif ]] && mrconvert T1.mif T1.nii.gz -force; }
  [[ -f T1.nii.gz ]] || { echo "[Skip] $SUBJ no T1"; cd ..; continue; }

  # 1) MNI→T1, 2) Apply atlas, 3) T1→DWI NN
  flirt -in "$MNI_T1" -ref T1.nii.gz -omat MNItoT1.mat -dof 12 -cost corratio
  flirt -in "$ATLAS_MNI" -ref T1.nii.gz -applyxfm -init MNItoT1.mat \
        -interp nearestneighbour -out atlas_in_T1.nii.gz
  mrconvert atlas_in_T1.nii.gz atlas_in_T1.mif -force -quiet
  transformcalc diff2struct_mrtrix.txt invert struct2diff.txt -force
  mrtransform atlas_in_T1.mif -linear struct2diff.txt \
             -template mean_b0.mif -interp nearest atlas_in_dwi.mif -force -quiet
  mrcalc atlas_in_dwi.mif 0.5 -add -floor -datatype uint16 atlas_in_dwi_uint16.mif -force

  # 4) Rebuild connectome if TCK+weights exist
  TCK=${SUBJ}_filtered_v2.tck
  WTS=${SUBJ}_sift_filtered_v2.txt
  if [[ -f "$TCK" && -f "$WTS" ]]; then
    mkdir -p connectome_local
    tck2connectome "$TCK" atlas_in_dwi_uint16.mif \
      connectome_local/${SUBJ}_SC_sift2.csv \
      -tck_weights_in "$WTS" -symmetric -zero_diagonal \
      -assignment_radial_search 3 -force
  fi

  cd ..
done
echo "[Batch] Done."
