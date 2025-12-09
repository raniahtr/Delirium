# Delirium Connectome Analysis

Structural and functional connectome analysis pipeline for delirium research.

## Overview

This repository contains analysis scripts and utilities for processing and analyzing brain connectomes from diffusion MRI (dMRI) and functional MRI (fMRI) data. The pipeline includes preprocessing, connectome construction, and statistical analysis tools.

## Repository Structure

```
.
├── scripts/              # Analysis scripts and utilities
│   ├── utils/           # Common utilities (connectome loading, plotting, etc.)
│   ├── notebooks/       # Jupyter notebooks for analysis
│   └── *.py            # Standalone analysis scripts
├── atlas/              # Brain atlas files and mappings
│   ├── final_atlas_to_yeo7.csv
│   └── parcels_label.xlsx
├── Preproc_current/    # Preprocessed data (not in git - too large)
└── rawdata_current/    # Raw data (not in git - too large)
```

## Setup

### Prerequisites

- Python 3.7+
- Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `nibabel`
- Jupyter Notebook (for interactive analysis)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Delirium_Rania
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure paths:
   - Update `ROOT_DIR` in `scripts/utils/connectome_utils.py` to point to your data directory
   - Or set environment variable: `export DELIRIUM_ROOT_DIR=/path/to/your/data`

## Usage

### Quick Start

```python
from scripts.utils import (
    load_all_connectomes,
    compute_group_average,
    PREPROC_DIR
)

# Load connectomes
connectomes = load_all_connectomes(
    preproc_dir=PREPROC_DIR,
    connectome_type='SC_sift2_sizecorr'
)

# Compute group average
avg_connectome = compute_group_average(connectomes)
```

### Main Analysis Notebooks

- `scripts/notebooks/new_analysis.ipynb` - Main structural connectome analysis
- `scripts/notebooks/dwi_preprocessing_qc.ipynb` - DWI preprocessing QC
- `scripts/notebooks/qc_pattern_analysis.ipynb` - QC pattern analysis

### Key Scripts

- `scripts/regenerate_connectomes_430.py` - Regenerate connectomes with 430-parcel atlas
- `scripts/sizecorr.py` - Size correction for connectomes
- `scripts/snr_voxelwise_qc.py` - SNR voxelwise QC analysis

## Atlas Information

- **Total ROIs**: 430 (after removing parcels 390 and 423)
- **Cortical**: Parcels 1-360 (Yeo7 networks)
- **Subcortical**: Parcels 361-424
- **Brainstem**: Parcels 425-428
- **Cerebellum**: Parcels 429-430 (excluded from similarity analysis due to insufficient edges)

## Data Organization

Data should be organized as:
```
Preproc_current/
├── sub-XX/
│   └── connectome_local/
│       ├── sub-XX_SC_sift2.csv
│       ├── sub-XX_SC_sift2_sizecorr.csv
│       ├── sub-XX_MEAN_FA.csv
│       └── sub-XX_MEAN_MD.csv
```

## Features

- **Connectome Loading**: Automatic caching for efficient data loading
- **Group Analysis**: Built-in functions for group comparisons
- **Visualization**: Custom plotting functions for connectomes
- **QC Tools**: SNR and preprocessing quality control
- **Statistical Analysis**: Connectivity strength, efficiency, and similarity metrics

## Configuration

Key configuration is in `scripts/utils/connectome_utils.py`:

- `ROOT_DIR`: Root project directory
- `PREPROC_DIR`: Preprocessing directory
- `ATLAS_DIR`: Atlas files directory
- `N_ROIS`: Number of ROIs (430)

## Contributing

1. Create a feature branch
2. Make your changes
3. Test with sample data
4. Submit a pull request

## License

[Add your license here]

## Citation

If you use this code, please cite:
[Add citation information]

## Contact

[Add contact information]

