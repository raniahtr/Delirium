# Data Requirements and Organization

This document describes the data requirements for running the connectome analysis pipeline.

## Data Directory Structure

The pipeline expects data to be organized as follows:

```
<ROOT_DIR>/
├── Preproc_current/          # Preprocessed data directory
│   ├── sub-XX/              # Subject directories
│   │   └── connectome_local/
│   │       ├── sub-XX_SC_sift2.csv
│   │       ├── sub-XX_SC_sift2_sizecorr.csv
│   │       ├── sub-XX_MEAN_FA.csv
│   │       └── sub-XX_MEAN_MD.csv
│   └── results/             # Analysis results (optional)
│
├── atlas/                    # Atlas files
│   ├── final_atlas_to_yeo7.csv
│   ├── parcels_label.xlsx
│   └── [other atlas files]
│
└── rawdata_current/          # Raw data (optional, not used by analysis)
```

## Setting Up Data

### Option 1: Use Environment Variable

Set the root directory using an environment variable:

```bash
export DELIRIUM_ROOT_DIR=/path/to/your/data
```

### Option 2: Modify Configuration

Edit `scripts/utils/connectome_utils.py` and update `ROOT_DIR`:

```python
ROOT_DIR = Path("/path/to/your/data")
```

## Connectome File Format

Connectome CSV files should be:
- **Format**: Comma-separated values (CSV)
- **Size**: 430×430 matrices (after removing parcels 390 and 423)
- **Content**: Connectivity strength values 
- **Naming**: `sub-<SUBJECT_ID>_<CONNECTOME_TYPE>.csv`

### Connectome Types

- `SC_sift2`: Structural connectome (SIFT2)
- `SC_sift2_sizecorr`: Size-corrected structural connectome
- `MEAN_FA`: Mean fractional anisotropy
- `MEAN_MD`: Mean diffusivity

## Atlas Files

Required atlas files:
- `atlas/final_atlas_to_yeo7.csv`: Mapping of parcels to Yeo7 networks
- `atlas/parcels_label.xlsx`: Detailed parcel labels

Optional (for visualization):
- Atlas NIfTI files (`.nii.gz`)
- Yeo7 network atlas files

## Data Access

**Note**: The actual data files (connectomes, NIfTI files) are not included in this repository due to size and privacy considerations. 


