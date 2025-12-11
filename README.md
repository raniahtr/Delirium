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
├── SDI/                # Structural Decoupling Index analysis
│   ├── data/          # Data for Structural Decoupling Index analysis
│   ├── GSP_StructuralDecouplingIndex/ # Code for Structural Decoupling Index analysis (from paper)
│   └── spm/           # SPM software
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

## Atlas Information

- **Total ROIs**: 430  
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


## Configuration

Key configuration is in `scripts/utils/connectome_utils.py`:

- `ROOT_DIR`: Root project directory
- `PREPROC_DIR`: Preprocessing directory
- `ATLAS_DIR`: Atlas files directory
- `N_ROIS`: Number of ROIs (430)

