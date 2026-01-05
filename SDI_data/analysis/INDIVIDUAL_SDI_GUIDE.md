# Guide: Running Analysis with Individual Subject SDI Data

## Current State (Group-Averaged SDI)

**Current Data Format:**
- Files: `SDI_CR_allFB.h5`, `SDI_Delirium_allFB.h5`, `SDI_HC_allFB.h5`
- Each file contains: **430 SDI values** (one per parcel, **group-averaged**)
- **Limitation**: Cannot perform valid statistical tests - only descriptive analysis

**Current Analysis:**
- Descriptive statistics (mean, median, std, quartiles)
- Visualizations (distributions, brain projections, heatmaps)
- **Statistical tests are invalid** because they compare single values per group

## Required State (Individual Subject SDI)

**Required Data Format:**
- Files: `SDI_sub-AD_allFB.h5`, `SDI_sub-AF_allFB.h5`, etc. (one per subject)
- Each file contains: **430 SDI values** (one per parcel, for that specific subject)
- **Enables**: Valid statistical tests with proper sample sizes

**Expected Structure:**
```
SDI/
├── SDI_sub-AD_allFB.h5    # Subject AD: 430 SDI values
├── SDI_sub-AF_allFB.h5    # Subject AF: 430 SDI values
├── SDI_sub-AG_allFB.h5    # Subject AG: 430 SDI values
└── ...
```

Each `.h5` file should contain:
- A dataset named `SDI` with shape `(430,)` - one SDI value per parcel
- Or a pandas DataFrame with index as parcel IDs (0-429) and a column `SDI`

## Implementation Steps

### Step 1: Modify SDI Calculation

**File:** `SDI/calculate_SDI.py` (or wherever SDI is calculated)

Create a new function to calculate SDI per subject:

```python
def calculate_SDI_per_subject(
    subject_id: str,
    timeseries_file: str,
    structural_connectome: np.ndarray,
    output_dir: str,
    band: str = 'allFB'
):
    """
    Calculate SDI for a single subject.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'AD', 'AF')
    timeseries_file : str
        Path to subject's fMRI timeseries file
    structural_connectome : np.ndarray
        Structural connectome matrix (430 × 430)
    output_dir : str
        Directory to save SDI file
    band : str
        Frequency band identifier
    """
    # Load subject-specific timeseries
    # ... (load timeseries for this subject)
    
    # Calculate SDI using the same methodology as group-averaged version
    # ... (SDI calculation code)
    
    # Save as h5 file
    output_path = Path(output_dir) / f"SDI_sub-{subject_id}_{band}.h5"
    sdi_df = pd.DataFrame({'SDI': sdi_values}, index=range(430))
    sdi_df.to_hdf(str(output_path), key='SDI', mode='w')
    
    return output_path
```

**Key Changes:**
- Process each subject individually
- Use subject-specific timeseries (not group-averaged)
- Save one file per subject

### Step 2: Modify Data Loading

**File:** `analysis/load_sdi_data.py`

Add a new function to load individual subject SDI data:

```python
def load_individual_sdi_data(
    config: Dict,
    band: str = 'allFB',
    subject_group_mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load individual subject SDI data and create unified DataFrame.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    band : str
        Frequency band identifier
    subject_group_mapping : dict, optional
        Mapping from subject_id to group name.
        If None, will try to infer from file names or config.
        Format: {'AD': 'CR', 'AF': 'Delirium', ...}
    
    Returns:
    --------
    unified_df : pd.DataFrame
        DataFrame with columns: [subject_id, parcel_id, sdi_value, group]
    group_arrays : dict
        Dictionary mapping group names to arrays (n_subjects × 430)
    """
    sdi_dir = Path(config['sdi_dir']) / band
    
    # Find all subject SDI files
    sdi_files = sorted(sdi_dir.glob(f"SDI_sub-*_{band}.h5"))
    
    if len(sdi_files) == 0:
        raise ValueError(f"No individual SDI files found in {sdi_dir}")
    
    # Load subject-group mapping
    if subject_group_mapping is None:
        # Try to load from config or infer from file names
        subject_group_mapping = _infer_subject_groups(config, sdi_files)
    
    # Load all subject SDI data
    all_data = []
    for sdi_file in sdi_files:
        # Extract subject ID from filename
        subject_id = sdi_file.stem.split('_')[0].replace('SDI_sub-', '')
        group = subject_group_mapping.get(subject_id, 'Unknown')
        
        # Load SDI values
        sdi_df = pd.read_hdf(str(sdi_file), key='SDI')
        sdi_values = sdi_df['SDI'].values if 'SDI' in sdi_df.columns else sdi_df.values.flatten()
        
        # Create DataFrame for this subject
        subject_df = pd.DataFrame({
            'subject_id': subject_id,
            'parcel_id': range(len(sdi_values)),
            'sdi_value': sdi_values,
            'group': group
        })
        all_data.append(subject_df)
    
    # Combine all subjects
    unified_df = pd.concat(all_data, ignore_index=True)
    
    # Create group arrays (for backward compatibility)
    group_arrays = {}
    for group in unified_df['group'].unique():
        group_data = unified_df[unified_df['group'] == group]
        # Reshape to (n_subjects, 430)
        n_subjects = len(group_data['subject_id'].unique())
        group_array = group_data['sdi_value'].values.reshape(n_subjects, 430)
        group_arrays[group] = group_array
    
    return unified_df, group_arrays


def _infer_subject_groups(config: Dict, sdi_files: List[Path]) -> Dict[str, str]:
    """
    Infer subject-group mapping from config or file structure.
    
    This is a helper function - you may need to customize based on your
    file naming conventions or create a separate mapping file.
    """
    # Option 1: Load from config
    if 'subject_group_mapping' in config:
        return config['subject_group_mapping']
    
    # Option 2: Load from separate CSV file
    mapping_file = Path(config.get('subject_group_mapping_file', ''))
    if mapping_file.exists():
        mapping_df = pd.read_csv(mapping_file)
        return dict(zip(mapping_df['subject_id'], mapping_df['group']))
    
    # Option 3: Infer from directory structure
    # (e.g., if files are in subdirectories like CR/, Delirium/, HC/)
    # ... (custom logic)
    
    # Default: return empty dict (will result in 'Unknown' group)
    return {}
```

**Key Changes:**
- Load multiple files (one per subject)
- Create DataFrame with `subject_id` column
- Each parcel now has multiple values per group (one per subject)
- Maintain backward compatibility with `group_arrays` format

### Step 3: Update Pipeline

**File:** `analysis/run_full_analysis.py`

Modify the main function to support both data types:

```python
def run_full_analysis(
    config: Dict,
    band: str = 'allFB',
    use_individual_subjects: bool = False,
    output_dir: Optional[str] = None,
    ...
):
    """
    Run full SDI analysis pipeline.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    band : str
        Frequency band identifier
    use_individual_subjects : bool
        If True, load individual subject SDI data.
        If False, load group-averaged SDI data (descriptive only).
    output_dir : str, optional
        Output directory
    ...
    """
    # Load data
    if use_individual_subjects:
        from .load_sdi_data import load_individual_sdi_data
        unified_df, group_arrays = load_individual_sdi_data(config, band=band)
        logger.info(f"Loaded individual SDI data: {len(unified_df)} rows")
        logger.info(f"  Subjects per group: {unified_df.groupby('group')['subject_id'].nunique().to_dict()}")
    else:
        from .load_sdi_data import load_all_sdi_data
        unified_df, group_arrays = load_all_sdi_data(config, band=band)
        logger.warning("Using group-averaged SDI data - statistical tests will be invalid!")
        logger.warning("  Only descriptive analysis will be performed.")
    
    # Rest of the pipeline remains the same...
    # Statistical tests will now work correctly with individual subject data
```

### Step 4: Create Subject-Group Mapping File

Create a CSV file mapping subjects to groups:

**File:** `data/subject_group_mapping.csv`

```csv
subject_id,group
AD,CR
AF,Delirium
AG,HC
...
```

Or add to config file:

```python
config = {
    'sdi_dir': '/path/to/SDI',
    'subject_group_mapping': {
        'AD': 'CR',
        'AF': 'Delirium',
        'AG': 'HC',
        ...
    }
}
```

## Running the Analysis

### With Individual Subject Data

```python
from analysis.run_full_analysis import run_full_analysis
from utils.load_config import load_config

config = load_config()

# Run with individual subject data
run_full_analysis(
    config=config,
    band='allFB',
    use_individual_subjects=True,  # Enable individual subject analysis
    output_dir='results/analysis/individual_subjects'
)
```

### Command Line

```bash
cd /media/RCPNAS/Data/Delirium/Delirium_Rania/github/SDI_delirium
python -m analysis.run_full_analysis \
    --band allFB \
    --use-individual-subjects \
    --output-dir results/analysis/individual_subjects
```

## Expected Results

### With Individual Subject Data

**Valid Statistical Tests:**
- Group-level comparisons: Kruskal-Wallis, ANOVA (with proper sample sizes)
- Parcel-level comparisons: Multiple comparisons correction (FDR, Bonferroni)
- Post-hoc tests: Mann-Whitney U, Welch's t-test (with proper sample sizes)
- Effect sizes: Cohen's d, confidence intervals (with proper sample sizes)

**Data Structure:**
- `unified_df`: Each row is one parcel for one subject
  - Example: 430 parcels × 20 subjects = 8,600 rows
- `group_arrays`: Each group has shape (n_subjects, 430)
  - Example: CR group with 10 subjects → (10, 430) array

**Visualizations:**
- All plots will show proper distributions (not single values)
- Boxplots will show IQR and outliers correctly
- Statistical annotations will be meaningful

### With Group-Averaged Data (Current)

**Limitations:**
- Statistical tests are **invalid** (comparing single values)
- Only descriptive statistics are meaningful
- Visualizations show single values per group (not distributions)

## Verification

After implementing individual subject loading, verify:

1. **Data Structure:**
   ```python
   print(f"Total rows: {len(unified_df)}")
   print(f"Expected: {n_parcels} parcels × {n_subjects} subjects = {n_parcels * n_subjects}")
   print(f"Subjects per group:")
   print(unified_df.groupby('group')['subject_id'].nunique())
   ```

2. **Statistical Tests:**
   - Check that p-values are not all identical
   - Verify sample sizes in test results match actual subject counts
   - Confirm effect sizes are reasonable

3. **Visualizations:**
   - Boxplots should show distributions (not single points)
   - Violin plots should show proper density curves
   - Error bars should be visible and meaningful

## Troubleshooting

**Issue: "No individual SDI files found"**
- Check that files are named correctly: `SDI_sub-XXX_allFB.h5`
- Verify `sdi_dir` in config points to correct directory
- Check that files are in the `band` subdirectory

**Issue: "All p-values are identical"**
- This indicates group-averaged data is still being used
- Verify `use_individual_subjects=True` is set
- Check that `load_individual_sdi_data` is being called

**Issue: "Subject-group mapping not found"**
- Create `subject_group_mapping.csv` file
- Or add mapping to config file
- Or implement custom inference logic in `_infer_subject_groups`

## Next Steps

1. **Calculate Individual SDI Values:**
   - Modify SDI calculation script to process each subject
   - Generate `SDI_sub-XXX_allFB.h5` files

2. **Create Subject-Group Mapping:**
   - Create CSV file or add to config
   - Verify all subjects are mapped correctly

3. **Test Data Loading:**
   - Run `load_individual_sdi_data` function
   - Verify DataFrame structure and group assignments

4. **Run Full Analysis:**
   - Execute pipeline with `use_individual_subjects=True`
   - Verify statistical tests produce valid results
   - Check that visualizations show proper distributions




