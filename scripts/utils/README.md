# Connectome Analysis Utilities

This module provides common utilities for connectome analysis, extracted from recurring patterns in analysis notebooks.

## Quick Start

```python
from scripts.utils import (
    # Configuration paths
    PREPROC_DIR_V2, ATLAS_DIR, N_ROIS,
    
    # Subject utilities
    find_subjects, find_subjects_with_connectomes,
    
    # Connectome loading
    load_connectome, load_all_connectomes, compute_group_average,
    
    # Plotting
    setup_plotting_style, plot_log_connectome, plot_all_connectomes,
    
    # QC data
    load_qc_data, extract_qc_metrics,
)

# Setup plotting
setup_plotting_style()

# Find subjects
subjects = find_subjects(PREPROC_DIR_V2)

# Load connectomes
connectomes = load_all_connectomes(PREPROC_DIR_V2, connectome_type='SC_sift2')

# Compute group average
avg_connectome = compute_group_average(connectomes)

# Plot
plot_log_connectome(avg_connectome, 'Group Average')
```

## Configuration

All paths are centralized in `connectome_utils.py`:

- `ROOT_DIR`: Main project directory
- `PREPROC_DIR_V2`: Current preprocessing directory
- `ATLAS_DIR`: Atlas files directory
- `N_ROIS`: Number of ROIs (432)

## Common Use Cases

### 1. Load Connectomes for All Subjects

```python
from scripts.utils import load_all_connectomes, PREPROC_DIR_V2

connectomes = load_all_connectomes(
    preproc_dir=PREPROC_DIR_V2,
    connectome_type='SC_sift2_sizecorr'
)
```

### 2. Iterate Over Subjects

```python
from scripts.utils import iterate_subjects, get_connectome_path, load_connectome

subjects = iterate_subjects(PREPROC_DIR_V2)

for sub_id in subjects:
    connectome_path = get_connectome_path(sub_id, 'SC_sift2', PREPROC_DIR_V2)
    if connectome_path:
        matrix = load_connectome(connectome_path)
        # Do analysis...
```

### 3. Load QC Data

```python
from scripts.utils import load_qc_data, extract_qc_metrics, QC_QUALITATIVE_JSON

qc_data = load_qc_data(QC_QUALITATIVE_JSON)
df = extract_qc_metrics(qc_data)
```

### 4. Plot Connectomes

```python
from scripts.utils import plot_all_connectomes, setup_plotting_style

setup_plotting_style()
fig, axes = plot_all_connectomes(
    connectomes,
    max_subjects=20,
    n_cols=4,
    connectome_type='SC_sift2'
)
```

### 5. Group Comparisons

```python
from scripts.utils import plot_group_comparison_boxplot

# Assuming you have a DataFrame with 'group' and metric columns
fig, ax = plot_group_comparison_boxplot(
    df,
    metric='fa_mean',
    group_col='group',
    groups=['Group1', 'Group2', 'Group3'],
    colors=['#E67E22', '#C0392B', '#3498DB']  # Orange/red shades + blue
)
```





