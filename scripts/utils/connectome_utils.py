"""
Common utilities for connectome analysis and processing.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from itertools import combinations
import glob
import json
import warnings
import random
warnings.filterwarnings('ignore')
import matplotlib as mpl


# ============================================================================
# CONFIGURATION PATHS
# ============================================================================

# Root directory - can be overridden with environment variable
ROOT_DIR = Path(os.getenv('DELIRIUM_ROOT_DIR', '/media/RCPNAS/Data/Delirium/Delirium_Rania'))

# Preprocessing directories
PREPROC_DIR = ROOT_DIR / "Preproc_current"


# Atlas directory
ATLAS_DIR = ROOT_DIR / "atlas"
FINAL_ATLAS_PATH = ATLAS_DIR / "Final_Combined_atlas_MNI2009c_1mm.nii.gz"
PARCELS_LABELS_XLSX = ATLAS_DIR / "parcels_label.xlsx"
YEO7_PATH = ATLAS_DIR / "Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz"
YEO7_RESAMP_PATH = ATLAS_DIR / "yeo7_resamp_to_final_atlas.nii.gz"
ATLAS_TO_YEO7_CSV = ATLAS_DIR / "final_atlas_to_yeo7.csv"


# Number of ROIs in final atlas (after removing parcels 390 and 423)
N_ROIS = 430

# Parcels excluded from the atlas (original 1-indexed IDs)
EXCLUDED_PARCELS = [390, 423]


# ============================================================================
# PARCEL ID SHIFTING HELPERS
# ============================================================================

def shift_parcel_id(original_id: int) -> int:
    """
    Map original parcel ID (1-432, excluding 390 and 423) to shifted ID (1-430).
    
    After removing parcels 390 and 423, subsequent parcels are shifted down:
    - Parcels 1-389: unchanged (1-389)
    - Parcel 390: REMOVED
    - Parcels 391-422: shift down by 1 (390-421)
    - Parcel 423: REMOVED
    - Parcels 424-432: shift down by 2 (422-430)
    
    Parameters:
    -----------
    original_id : int
        Original parcel ID (1-432, excluding 390 and 423)
        
    Returns:
    --------
    int
        Shifted parcel ID (1-430)
    """
    if original_id in EXCLUDED_PARCELS:
        raise ValueError(f"Parcel {original_id} is excluded and cannot be shifted")
    
    if original_id < 1 or original_id > 432:
        raise ValueError(f"Parcel ID {original_id} is out of valid range (1-432)")
    
    if original_id < 390:
        # Parcels 1-389: unchanged
        return original_id
    elif original_id < 423:
        # Parcels 391-422: shift down by 1
        return original_id - 1
    else:
        # Parcels 424-432: shift down by 2
        return original_id - 2


def unshift_parcel_id(shifted_id: int) -> int:
    """
    Map shifted parcel ID (1-430) back to original ID (1-432, excluding 390 and 423).
    
    This is the inverse of shift_parcel_id().
    
    Parameters:
    -----------
    shifted_id : int
        Shifted parcel ID (1-430)
        
    Returns:
    --------
    int
        Original parcel ID (1-432, excluding 390 and 423)
    """
    if shifted_id < 1 or shifted_id > 430:
        raise ValueError(f"Shifted parcel ID {shifted_id} is out of valid range (1-430)")
    
    if shifted_id < 390:
        # Parcels 1-389: unchanged
        return shifted_id
    elif shifted_id < 422:
        # Parcels 390-421: shift up by 1 (originally 391-422)
        return shifted_id + 1
    else:
        # Parcels 422-430: shift up by 2 (originally 424-432)
        return shifted_id + 2


# Common connectome types
CONNECTOME_TYPES = {
    'SC_sift2': 'SC_sift2',
    'SC_sift2_sizecorr': 'SC_sift2_sizecorr',
    'COUNT': 'COUNT',
    'TOTAL_length': 'TOTAL_length',
    'MEAN_length': 'MEAN_length',
    'MEAN_FA': 'MEAN_FA',
    'MEAN_MD': 'MEAN_MD',
    'MEAN_AD': 'MEAN_AD',
    'MEAN_RD': 'MEAN_RD',
    'SC_invlen_sum': 'SC_invlen_sum'
}

# Default groups for analysis (standardized naming)
DEFAULT_GROUPS = {
    'ICU': ['AF', 'DA2', 'PM', 'BA', 'VC'],
    'ICU Delirium': ['CG', 'DA', 'FS', 'FSE', 'GL', 'KJ', 'LL', 'MF', 'PMA', 'PO', 'PB', 'SA'],
    'Healthy Controls': ['FEF', 'FD', 'GB', 'SG', 'AR', 'TL', 'TOG', 'PL', 'ZM', 'AM', 'PC', 'AD']
}


# ============================================================================
# SUBJECT UTILITIES
# ============================================================================

def find_subjects(preproc_dir: Union[str, Path] = PREPROC_DIR, 
                  pattern: str = "sub-*") -> List[str]:
    """
    Find all subject directories in preprocessing directory.
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Path to preprocessing directory
    pattern : str
        Pattern to match subject directories (default: "sub-*")
    
    Returns:
    --------
    subjects : list
        List of subject IDs (e.g., ['sub-AD', 'sub-AF', ...])
    """
    preproc_path = Path(preproc_dir)
    subjects = sorted([d.name for d in preproc_path.glob(pattern) if d.is_dir()])
    return subjects


def find_subjects_with_connectomes(preproc_dir: Union[str, Path] = PREPROC_DIR,
                                   connectome_type: str = "SC_sift2") -> Dict[str, Path]:
    """
    Find all subjects that have connectome_local folders with the specified connectome type.
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Path to preprocessing directory
    connectome_type : str
        Type of connectome to look for (e.g., "SC_sift2", "SC_sift2_sizecorr")
    
    Returns:
    --------
    subjects : dict
        Dictionary mapping subject IDs to their connectome file paths
    """
    preproc_path = Path(preproc_dir)
    subjects = {}
    
    for sub_dir in sorted(preproc_path.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        
        sub_id = sub_dir.name
        connectome_local = sub_dir / "connectome_local"
        
        if not connectome_local.exists():
            continue
        
        # Look for the connectome file
        connectome_file = connectome_local / f"{sub_id}_{connectome_type}.csv"
        
        if connectome_file.exists():
            subjects[sub_id] = connectome_file
        else:
            print(f"Warning: {sub_id} has connectome_local but missing {connectome_type}.csv")
    
    return subjects


def iterate_subjects(preproc_dir: Union[str, Path] = PREPROC_DIR,
                     subjects: Optional[List[str]] = None,
                     pattern: str = "sub-*") -> List[str]:
    """
    Get list of subjects to iterate over.
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Path to preprocessing directory
    subjects : list, optional
        Specific list of subjects to process. If None, finds all subjects.
    pattern : str
        Pattern to match subject directories if subjects is None
    
    Returns:
    --------
    subjects : list
        List of subject IDs to process
    """
    if subjects is None:
        subjects = find_subjects(preproc_dir, pattern)
    return subjects


# ============================================================================
# CONNECTOME CACHING
# ============================================================================

# Module-level cache for loaded connectomes
_connectome_cache: Dict[str, np.ndarray] = {}

# Cache configuration
CACHE_ENABLED = True


def enable_connectome_cache(enabled: bool = True) -> None:
    """
    Enable or disable connectome caching.
    
    Parameters:
    -----------
    enabled : bool
        If True, enable caching. If False, disable caching.
    """
    global CACHE_ENABLED
    CACHE_ENABLED = enabled


def clear_connectome_cache() -> None:
    """
    Clear all cached connectomes from memory.
    """
    global _connectome_cache
    _connectome_cache.clear()


def get_cache_info() -> Dict[str, Union[int, float, List[str]]]:
    """
    Get information about the connectome cache.
    
    Returns:
    --------
    info : dict
        Dictionary containing:
        - 'size': Number of cached connectomes
        - 'keys': List of cached file paths
        - 'memory_mb': Approximate memory usage in MB
        - 'enabled': Whether caching is enabled
    """
    global _connectome_cache, CACHE_ENABLED
    
    # Calculate approximate memory usage
    memory_bytes = sum(arr.nbytes for arr in _connectome_cache.values())
    memory_mb = memory_bytes / (1024 * 1024)
    
    return {
        'size': len(_connectome_cache),
        'keys': list(_connectome_cache.keys()),
        'memory_mb': memory_mb,
        'enabled': CACHE_ENABLED
    }


def preload_connectomes(preproc_dir: Union[str, Path] = PREPROC_DIR,
                        connectome_types: Optional[List[str]] = None,
                        subjects: Optional[List[str]] = None,
                        verbose: bool = True) -> Dict[str, int]:
    """
    Preload common connectome types into the cache.
    
    This function loads all connectomes of specified types for all (or specified) subjects
    into the cache, so subsequent operations are faster.
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Path to preprocessing directory
    connectome_types : list, optional
        List of connectome types to preload. If None, uses common defaults:
        ['SC_sift2_sizecorr', 'SC_sift2', 'MEAN_FA', 'MEAN_MD']
    subjects : list, optional
        Specific subjects to preload. If None, loads all available subjects.
    verbose : bool
        If True, print progress information
        
    Returns:
    --------
    summary : dict
        Dictionary with counts of loaded connectomes per type
    """
    if connectome_types is None:
        connectome_types = ['SC_sift2_sizecorr', 'SC_sift2', 'MEAN_FA', 'MEAN_MD']
    
    preproc_path = Path(preproc_dir)
    summary = {}
    
    if verbose:
        print("=" * 60)
        print("PRELOADING CONNECTOMES INTO CACHE")
        print("=" * 60)
        print(f"Connectome types: {', '.join(connectome_types)}")
        if subjects:
            print(f"Subjects: {', '.join(subjects)}")
        else:
            print("Subjects: All available")
        print()
    
    for conn_type in connectome_types:
        try:
            # Use load_all_connectomes which will cache automatically
            connectomes = load_all_connectomes(
                preproc_dir=preproc_path,
                connectome_type=conn_type,
                subjects=subjects
            )
            summary[conn_type] = len(connectomes)
            if verbose:
                print(f"✓ Preloaded {len(connectomes)} {conn_type} connectomes")
        except Exception as e:
            summary[conn_type] = 0
            if verbose:
                print(f"✗ Failed to preload {conn_type}: {e}")
    
    if verbose:
        cache_info = get_cache_info()
        print()
        print("=" * 60)
        print("PRELOAD SUMMARY")
        print("=" * 60)
        print(f"Total cached connectomes: {cache_info['size']}")
        print(f"Memory usage: {cache_info['memory_mb']:.2f} MB")
        print("=" * 60)
    
    return summary


# ============================================================================
# CONNECTOME LOADING FUNCTIONS
# ============================================================================

def load_connectome(connectome_path: Union[str, Path]) -> np.ndarray:
    """
    Load a connectome matrix from CSV file.
    
    Expects 430×430 matrices (after removal of parcels 390 and 423).
    If a 432×432 matrix is encountered, it will be automatically converted
    by removing rows/columns 389 and 422 (0-indexed).
    
    Uses caching to avoid reloading the same file multiple times.
    Caching can be controlled via enable_connectome_cache() and cleared
    via clear_connectome_cache().
    
    Parameters:
    -----------
    connectome_path : str or Path
        Path to the connectome CSV file
    
    Returns:
    --------
    matrix : np.ndarray
        The connectome matrix (430×430). Returns a copy to prevent
        accidental mutations of cached data.
    """
    global _connectome_cache, CACHE_ENABLED
    
    # Normalize path to handle symlinks and relative paths
    connectome_path = Path(connectome_path).resolve()
    cache_key = str(connectome_path)
    
    # Check cache first if enabled
    if CACHE_ENABLED and cache_key in _connectome_cache:
        # Return a copy to prevent accidental mutations
        return _connectome_cache[cache_key].copy()
    
    # Cache miss or caching disabled - load from disk
    try:
        # Use file handle approach for robustness
        with open(connectome_path, 'r') as f:
            matrix = np.loadtxt(f, delimiter=",")
        
        # Handle legacy 432×432 matrices by removing excluded parcels
        if matrix.shape == (432, 432):
            # Remove rows/columns 389 and 422 (0-indexed for parcels 390 and 423)
            excluded_indices = [389, 422]
            keep_indices = [i for i in range(432) if i not in excluded_indices]
            matrix = matrix[np.ix_(keep_indices, keep_indices)]
            print(f"Warning: Converted 432×432 to 430×430 for {connectome_path.name}")
        elif matrix.shape != (430, 430):
            raise ValueError(f"Expected 430×430 or 432×432 matrix, got {matrix.shape}")
        
        # Use prepare_matrix() to ensure symmetric, zero diagonal, and handle NaNs
        matrix = prepare_matrix(matrix)
        
        # Store in cache if enabled
        if CACHE_ENABLED:
            _connectome_cache[cache_key] = matrix
        
        # Return a copy to prevent accidental mutations
        return matrix.copy()
    except Exception as e:
        print(f"Error loading {connectome_path}: {e}")
        raise


def load_all_connectomes(preproc_dir: Union[str, Path] = PREPROC_DIR,
                         connectome_type: str = "SC_sift2",
                         subjects: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Load connectomes for all subjects.
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Path to preprocessing directory
    connectome_type : str
        Type of connectome to load
    subjects : list, optional
        Specific subjects to load. If None, loads all available.
    
    Returns:
    --------
    connectomes : dict
        Dictionary mapping subject IDs to connectome matrices
    """
    preproc_path = Path(preproc_dir)
    connectomes = {}
    
    # Find subjects with connectomes
    subjects_dict = find_subjects_with_connectomes(preproc_path, connectome_type)
    
    # Filter to requested subjects if provided
    if subjects is not None:
        subjects_dict = {k: v for k, v in subjects_dict.items() if k in subjects}
    
    # Load each connectome
    for sub_id, connectome_path in subjects_dict.items():
        try:
            matrix = load_connectome(connectome_path)
            connectomes[sub_id] = matrix
            print(f"✓ Loaded {sub_id}: shape {matrix.shape}")
        except Exception as e:
            print(f"✗ Failed to load {sub_id}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(connectomes)} connectomes")
    return connectomes


def compute_group_average(connectomes_dict: Dict[str, np.ndarray],
                          subject_list: Optional[List[str]] = None) -> np.ndarray:
    """
    Compute the average connectome across a group of subjects.
    
    Parameters:
    -----------
    connectomes_dict : dict
        Dictionary mapping subject IDs to connectome matrices
    subject_list : list, optional
        Specific subjects to average. If None, averages all.
    
    Returns:
    --------
    average_connectome : np.ndarray
        Average connectome matrix
    """
    if subject_list is None:
        subject_list = list(connectomes_dict.keys())
    
    valid_subjects = [s for s in subject_list if s in connectomes_dict]
    
    if len(valid_subjects) == 0:
        raise ValueError("No valid subjects found in connectomes_dict")
    
    # Get all matrices
    matrices = [connectomes_dict[s] for s in valid_subjects]
    
    # Check shape consistency
    shapes = [m.shape for m in matrices]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"Inconsistent matrix shapes: {shapes}")
    
    # Compute average
    average = np.mean(np.array(matrices), axis=0)
    average = np.nan_to_num(average, nan=0.0, posinf=0.0, neginf=0.0)
    
    return average


# ============================================================================
# CONNECTOME PATTERN FINDING
# ============================================================================

def find_connectome_files(data_dir: Union[str, Path],
                          pattern: str = "*_SC_sift2.csv",
                          recursive: bool = True) -> List[Path]:
    """
    Find connectome files matching a pattern.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory to search in
    pattern : str
        Glob pattern to match (e.g., "*_SC_sift2.csv")
    recursive : bool
        Whether to search recursively
    
    Returns:
    --------
    files : list
        List of matching file paths
    """
    data_path = Path(data_dir)
    
    if recursive:
        files = sorted(data_path.glob(f"**/{pattern}"))
    else:
        files = sorted(data_path.glob(pattern))
    
    return files


def get_connectome_path(subject_id: str,
                        connectome_type: str = "SC_sift2",
                        preproc_dir: Union[str, Path] = PREPROC_DIR) -> Optional[Path]:
    """
    Get path to a specific subject's connectome file.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., "sub-AD")
    connectome_type : str
        Type of connectome
    preproc_dir : str or Path
        Preprocessing directory
    
    Returns:
    --------
    path : Path or None
        Path to connectome file, or None if not found
    """
    preproc_path = Path(preproc_dir)
    connectome_path = preproc_path / subject_id / "connectome_local" / f"{subject_id}_{connectome_type}.csv"
    
    if connectome_path.exists():
        return connectome_path
    return None


# SNR 
def load_snr_qc_data(preproc_dir):
    """
    Load SNR QC summary data from all subjects.
    
    Parameters:
    -----------
    preproc_dir : Path
        Path to Preproc_current directory
    
    Returns:
    --------
    snr_data : pd.DataFrame
        DataFrame with SNR metrics for all subjects
    """
    snr_records = []
    
    # Find all subjects with qc_snr directories
    for sub_dir in sorted(preproc_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        
        sub_id = sub_dir.name
        qc_snr_dir = sub_dir / "qc_snr"
        
        # Check for summary CSV file
        summary_csv = qc_snr_dir / f"{sub_id}_snr_summary.csv"
        stats_json = qc_snr_dir / f"{sub_id}_snr_comparison_stats.json"
        
        if summary_csv.exists():
            try:
                # Load CSV summary
                df_summary = pd.read_csv(summary_csv)
                
                # Extract key metrics
                record = {'subject': sub_id}
                
                # Get mean SNR values
                mean_row = df_summary[df_summary['metric'] == 'mean']
                if len(mean_row) > 0:
                    record['snr_pre'] = mean_row.iloc[0]['preprocessing']
                    record['snr_post'] = mean_row.iloc[0]['postprocessing']
                    record['snr_improvement'] = mean_row.iloc[0]['improvement']
                    record['snr_improvement_pct'] = mean_row.iloc[0]['improvement_pct']
                
                # Get median
                median_row = df_summary[df_summary['metric'] == 'median']
                if len(median_row) > 0:
                    record['snr_pre_median'] = median_row.iloc[0]['preprocessing']
                    record['snr_post_median'] = median_row.iloc[0]['postprocessing']
                
                # Get std
                std_row = df_summary[df_summary['metric'] == 'std']
                if len(std_row) > 0:
                    record['snr_pre_std'] = std_row.iloc[0]['preprocessing']
                    record['snr_post_std'] = std_row.iloc[0]['postprocessing']
                
                # Load JSON for additional details
                if stats_json.exists():
                    with open(stats_json, 'r') as f:
                        stats_data = json.load(f)
                        record['noise_std_global'] = stats_data.get('noise_std_global', np.nan)
                        record['correlation'] = stats_data['statistics']['improvement'].get('correlation', np.nan)
                        record['rmse'] = stats_data['statistics']['improvement'].get('rmse', np.nan)
                        record['n_voxels_improved'] = stats_data['statistics']['voxel_counts'].get('improved', np.nan)
                        record['n_voxels_degraded'] = stats_data['statistics']['voxel_counts'].get('degraded', np.nan)
                        record['n_voxels_total'] = stats_data['statistics']['voxel_counts'].get('total_valid', np.nan)
                
                snr_records.append(record)
                
            except Exception as e:
                print(f"Warning: Failed to load SNR data for {sub_id}: {e}")
                continue
    
    if len(snr_records) == 0:
        print("No SNR QC data found")
        return pd.DataFrame()
    
    return pd.DataFrame(snr_records)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def get_group_cmap(group_name, default="viridis"):
    group_cmaps = {
    "ICU": mpl.colors.LinearSegmentedColormap.from_list(
        "icu_cmap",
        ["#fff3e5", "#f0a65a", "#E67E22", "#a34f0f"]
    ),
    "ICU Delirium": mpl.colors.LinearSegmentedColormap.from_list(
        "icu_delirium_cmap",
        ["#ffe9e5", "#d86a58", "#C0392B", "#7f1f14"]
    ),
    "Healthy Controls": mpl.colors.LinearSegmentedColormap.from_list(
        "healthy_cmap",
        ["#e8f3ff", "#6faee4", "#3498DB", "#1f5d8f"]
    ),
    }
    return group_cmaps.get(group_name, default)

def setup_plotting_style(style: str = 'whitegrid', dpi: int = 100, 
                        save_dpi: int = 300, palette: str = 'husl'):
    """
    Setup matplotlib and seaborn plotting style.
    
    Parameters:
    -----------
    style : str
        Seaborn style name
    dpi : int
        Display DPI
    save_dpi : int
        Save DPI
    palette : str
        Color palette name
    """
    # Try different style names for compatibility
    try:
        sns.set_style(style)
    except OSError:
        try:
            sns.set_style('seaborn-v0_8-darkgrid')
        except OSError:
            plt.style.use('dark_background')
            print("Note: Using 'dark_background' style (seaborn styles not available)")
    
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = save_dpi
    sns.set_palette(palette)


def plot_log_connectome(matrix: np.ndarray, subject_id: str, ax=None,
                       cmap: str = 'viridis', vmin: Optional[float] = None,
                       vmax: Optional[float] = None, title: Optional[str] = None):
    """
    Plot a log-transformed connectome matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        The connectome matrix
    subject_id : str
        Subject ID for title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str
        Colormap name
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    title : str, optional
        Plot title. If None, uses subject_id.
    """
    # Apply log1p transformation
    log_matrix = np.log1p(matrix)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(log_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xlabel('Target ROI', fontsize=10)
    ax.set_ylabel('Source ROI', fontsize=10)
    
    if title is None:
        title = f'{subject_id}\nLog Connectome (log1p)'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='log(1 + weight)')
    
    return ax


def plot_all_connectomes(connectomes_dict: Dict[str, np.ndarray],
                        max_subjects: Optional[int] = None,
                        n_cols: int = 4,
                        figsize_per_subject: Tuple[int, int] = (4, 4),
                        connectome_type: str = "SC_sift2"):
    """
    Plot all connectomes in a grid layout.
    
    Parameters:
    -----------
    connectomes_dict : dict
        Dictionary mapping subject IDs to connectome matrices
    max_subjects : int, optional
        Maximum number of subjects to plot. If None, plots all.
    n_cols : int
        Number of columns in grid
    figsize_per_subject : tuple
        Figure size per subject (width, height)
    connectome_type : str
        Connectome type for title
    """
    subjects_to_plot = list(connectomes_dict.keys())
    
    if max_subjects is not None:
        subjects_to_plot = subjects_to_plot[:max_subjects]
    
    n_subjects = len(subjects_to_plot)
    n_rows = int(np.ceil(n_subjects / n_cols))
    
    # Compute common vmin/vmax across all connectomes for consistent scaling
    all_log_values = []
    for matrix in connectomes_dict.values():
        log_matrix = np.log1p(matrix)
        all_log_values.extend(log_matrix[log_matrix > 0].flatten())
    
    vmin = np.percentile(all_log_values, 5) if len(all_log_values) > 0 else None
    vmax = np.percentile(all_log_values, 95) if len(all_log_values) > 0 else None
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols * figsize_per_subject[0],
                                    n_rows * figsize_per_subject[1]))
    
    if n_subjects == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    
    # Plot each connectome
    for idx, sub_id in enumerate(subjects_to_plot):
        ax = axes_flat[idx]
        matrix = connectomes_dict[sub_id]
        plot_log_connectome(matrix, sub_id, ax=ax, vmin=vmin, vmax=vmax)
    
    # Hide unused axes
    for idx in range(n_subjects, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle(f'Log Connectomes (log1p) - {connectome_type}\nAll Subjects (N={n_subjects})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


def plot_group_comparison_boxplot(data: pd.DataFrame,
                                 metric: str,
                                 group_col: str = 'group',
                                 groups: Optional[List[str]] = None,
                                 colors: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (8, 6),
                                 title: Optional[str] = None,
                                 ylabel: Optional[str] = None):
    """
    Create a boxplot comparing groups for a given metric.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with metric and group columns
    metric : str
        Column name for the metric to plot
    group_col : str
        Column name for group labels
    groups : list, optional
        Specific groups to plot. If None, uses all groups.
    colors : list, optional
        Colors for each group. If None, uses default palette.
    figsize : tuple
        Figure size
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label. If None, uses metric name.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    if groups is None:
        groups = sorted(data[group_col].unique())
    
    # Filter data
    plot_data = data[data[group_col].isin(groups)]
    
    # Default colors (orange/red shades + blue)
    if colors is None:
        colors = ['#E67E22', '#C0392B', '#3498DB']
        # Extend if more groups
        while len(colors) < len(groups):
            colors.append('#95A5A6')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    box_data = [plot_data[plot_data[group_col] == g][metric].dropna().values 
                for g in groups]
    
    bp = ax.boxplot(box_data, labels=groups, patch_artist=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors[:len(groups)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.set_xlabel('Group')
    ax.grid(axis='y', alpha=0.3)
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def get_group_colors(groups: List[str]) -> List[str]:
    """
    Get color mapping for groups based on standard color scheme.
    
    Color scheme:
    - ICU groups: Orange (#E67E22)
    - ICU Delirium groups: Darker red/orange (#C0392B)
    - Healthy Controls groups: Blue (#3498DB)
    
    Standard group names: "ICU", "ICU Delirium", "Healthy Controls"
    
    Parameters:
    -----------
    groups : list
        List of group names
    
    Returns:
    --------
    colors : list
        List of hex color codes corresponding to each group
    """
    # Standard color scheme: orange/red shades for ICU/delirium, blue for healthy
    color_map = {
        'icu': '#E67E22',           # Orange (for "ICU")
        'delirium': '#C0392B',      # Darker red/orange (for "ICU Delirium")
        'healthy': '#3498DB',       # Blue (for "Healthy Controls")
    }
    
    colors = []
    for group in groups:
        group_lower = group.lower().strip()
        # Handle standard names: "ICU", "ICU Delirium", "Healthy Controls"
        if group_lower == 'icu':
            colors.append(color_map['icu'])
        elif group_lower == 'icu delirium' or group_lower == 'icu_delirium' or group_lower == 'delirium icu' or group_lower == 'delirium_icu':
            colors.append(color_map['delirium'])
        elif group_lower == 'healthy controls' or group_lower == 'healthy_controls':
            colors.append(color_map['healthy'])
        # Handle variations with "group" prefix
        elif 'icu' in group_lower and 'delirium' not in group_lower:
            colors.append(color_map['icu'])
        elif 'delirium' in group_lower:
            colors.append(color_map['delirium'])
        elif 'healthy' in group_lower or 'control' in group_lower:
            colors.append(color_map['healthy'])
        else:
            # Fallback: use default palette order
            colors.append('#95A5A6')  # Gray
    
    return colors


def plot_connectome_heatmap(matrix: np.ndarray,
                           subject_id: Optional[str] = None,
                           ax=None,
                           cmap: str = 'viridis',
                           log_transform: bool = True,
                           vmin: Optional[float] = None,
                           vmax: Optional[float] = None,
                           title: Optional[str] = None):
    """
    Plot connectome as heatmap.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Connectome matrix
    subject_id : str, optional
        Subject ID for title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name
    log_transform : bool
        Whether to apply log1p transformation
    vmin, vmax : float, optional
        Value limits for colormap
    title : str, optional
        Plot title
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    plot_matrix = np.log1p(matrix) if log_transform else matrix
    
    im = ax.imshow(plot_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    if title is None and subject_id:
        title = f'{subject_id} Connectome'
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Target ROI', fontsize=10)
    ax.set_ylabel('Source ROI', fontsize=10)
    
    plt.colorbar(im, ax=ax, label='log(1 + weight)' if log_transform else 'weight')
    
    return ax


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def detect_outliers(series: pd.Series, name: str = "") -> Tuple[pd.Series, float, float]:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    series : pd.Series
        Data series
    name : str
        Name for reporting
    
    Returns:
    --------
    outliers : pd.Series
        Outlier values
    lower_bound : float
        Lower bound for normal range
    upper_bound : float
        Upper bound for normal range
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_connectome(matrix: np.ndarray,
                        normalize_by: str = 'mean',
                        target_density: Optional[float] = None,
                        log_transform: bool = False) -> np.ndarray:
    """
    Normalize connectome matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Connectome matrix
    normalize_by : str
        Normalization method: 'mean', 'max', 'sum', or 'density'
    target_density : float, optional
        Target density for density-based normalization
    log_transform : bool
        Whether to apply log1p transformation
    
    Returns:
    --------
    normalized : np.ndarray
        Normalized matrix
    """
    matrix = matrix.copy()
    
    if log_transform:
        matrix = np.log1p(matrix)
    
    if normalize_by == 'mean':
        mean_val = matrix[matrix > 0].mean()
        if mean_val > 0:
            matrix = matrix / mean_val
    elif normalize_by == 'max':
        max_val = matrix.max()
        if max_val > 0:
            matrix = matrix / max_val
    elif normalize_by == 'sum':
        sum_val = matrix.sum()
        if sum_val > 0:
            matrix = matrix / sum_val
    elif normalize_by == 'density' and target_density is not None:
        # Threshold to target density
        threshold = np.percentile(matrix[matrix > 0], 
                                 (1 - target_density) * 100)
        matrix[matrix < threshold] = 0
    
    return matrix


def prepare_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Prepare matrix for analysis: ensure symmetric, zero diagonal, handle NaNs.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix
    
    Returns:
    --------
    prepared : np.ndarray
        Prepared matrix
    """
    matrix = matrix.copy()
    
    # Ensure symmetric
    matrix = 0.5 * (matrix + matrix.T)
    
    # Zero diagonal
    np.fill_diagonal(matrix, 0.0)
    
    # Handle NaNs and infinities
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return matrix


# ============================================================================
# STRUCTURAL CONNECTOME SUBJECT SIMILARITY ANALYSIS
# ============================================================================

def get_region_category_indices(parcel_id: int, atlas_to_yeo7_df: pd.DataFrame) -> str:
    """
    Map a parcel ID to its region category.
    
    Region categories (after removal of parcels 390 and 423):
    - Yeo7_Network_1 through Yeo7_Network_7: Cortical parcels (1-360) mapped to Yeo7 networks
    - Subcortical: Parcels 361-424
    - Brainstem: Parcels 425-428
    - Cerebellum: Parcels 429-430
    
    Parameters:
    -----------
    parcel_id : int
        Parcel ID (1-430, shifted after removal of parcels 390 and 423)
    atlas_to_yeo7_df : pd.DataFrame
        DataFrame with columns 'parcel_id' and 'yeo7_network' from ATLAS_TO_YEO7_CSV
    
    Returns:
    --------
    region_category : str
        Region category name (e.g., 'Yeo7_Network_1', 'Subcortical', 'Brainstem', 'Cerebellum')
    """
    # Subcortical (361-424)
    if 361 <= parcel_id <= 424:
        return 'Subcortical'
    
    # Brainstem (425-428)
    if 425 <= parcel_id <= 428:
        return 'Brainstem'
    
    # Cerebellum (429-430)
    if 429 <= parcel_id <= 430:
        return 'Cerebellum'
    
    # Cortical parcels (1-360) - map to Yeo7 networks
    if 1 <= parcel_id <= 360:
        parcel_row = atlas_to_yeo7_df[atlas_to_yeo7_df['parcel_id'] == parcel_id]
        if len(parcel_row) > 0:
            yeo7_network = parcel_row['yeo7_network'].iloc[0]
            if pd.notna(yeo7_network) and yeo7_network > 0:
                return f'Yeo7_Network_{int(yeo7_network)}'
        # If no mapping found, return None (will be filtered out)
        return None
    
    # Unknown parcel ID
    return None


def extract_region_submatrix(connectome: np.ndarray, region_category: str, 
                             atlas_to_yeo7_df: pd.DataFrame) -> np.ndarray:
    """
    Extract submatrix from full connectome for a specific region category.
    
    For Yeo7 networks: extracts parcels belonging to that network.
    For subcortical/brainstem/cerebellum: extracts parcels in those ranges.
    Returns flattened upper triangle vector for similarity computation.
    
    Parameters:
    -----------
    connectome : np.ndarray
        Full connectome matrix (430x430, after removal of parcels 390 and 423)
    region_category : str
        Region category name (e.g., 'Yeo7_Network_1', 'Subcortical', 'Brainstem', 'Cerebellum')
    atlas_to_yeo7_df : pd.DataFrame
        DataFrame with columns 'parcel_id' and 'yeo7_network' from ATLAS_TO_YEO7_CSV
    
    Returns:
    --------
    region_vector : np.ndarray
        Flattened upper triangle vector of the region-specific submatrix
    """
    # Get parcel indices for this region category
    parcel_indices = []
    
    if region_category == 'Subcortical':
        parcel_indices = list(range(360, 424))  # 0-indexed: 360-423 (parcels 361-424)
    elif region_category == 'Brainstem':
        parcel_indices = list(range(424, 428))  # 0-indexed: 424-427 (parcels 425-428)
    elif region_category == 'Cerebellum':
        parcel_indices = list(range(428, 430))  # 0-indexed: 428-429 (parcels 429-430)
    elif region_category.startswith('Yeo7_Network_'):
        # Extract Yeo7 network number
        network_num = int(region_category.split('_')[-1])
        # Find all parcels belonging to this network
        network_parcels = atlas_to_yeo7_df[
            (atlas_to_yeo7_df['yeo7_network'] == network_num) & 
            (atlas_to_yeo7_df['parcel_id'] >= 1) & 
            (atlas_to_yeo7_df['parcel_id'] <= 360)
        ]['parcel_id'].values
        # Convert to 0-indexed
        parcel_indices = [int(pid - 1) for pid in network_parcels]
    else:
        raise ValueError(f"Unknown region category: {region_category}")
    
    if len(parcel_indices) == 0:
        # Return empty array if no parcels found
        return np.array([])
    
    # Extract submatrix
    submatrix = connectome[np.ix_(parcel_indices, parcel_indices)]
    
    # Skip cerebellum - it has only 2 parcels (1 edge), which is insufficient for meaningful correlation
    # Correlation requires at least 2 data points, and with only 1 edge, correlations would be undefined
    if region_category == 'Cerebellum':
        # Return empty array to signal that this region should be skipped
        return np.array([])
    
    # Get upper triangle (excluding diagonal) for all other regions
    triu_indices = np.triu_indices_from(submatrix, k=1)
    region_vector = submatrix[triu_indices]
    
    return region_vector


def compute_sc_similarity_between_subjects(
    preproc_dir: Union[str, Path],
    connectome_type: str = "SC_sift2",
    groups: Optional[Dict[str, List[str]]] = None,
    overwrite: bool = False,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compute Pearson correlation between subjects' connectomes for each region category.
    
    Similar to functional connectome similarity computation but for structural connectomes.
    Groups by region category (Yeo7 networks, subcortical, brainstem, cerebellum) instead of frequency band.
    Uses standardized group names: "ICU", "ICU Delirium", "Healthy Controls".
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Path to preprocessing directory
    connectome_type : str
        Type of connectome to use (default: "SC_sift2")
    groups : dict, optional
        Dictionary mapping group names to subject ID lists.
        If None, uses default groups: ICU, ICU Delirium, Healthy Controls
    overwrite : bool
        If False and save_path exists, load from file instead of recomputing
    save_path : Path, optional
        Path to save results (HDF5 or CSV). If None, saves to preproc_dir/results/similarity_matrices.h5
    
    Returns:
    --------
    similarity_df : pd.DataFrame
        DataFrame with columns: 'Group', 'Region Category', 'Subject i', 'Subject j', 'Similarity'
    """
    preproc_path = Path(preproc_dir)
    
    # Load atlas to Yeo7 mapping
    atlas_to_yeo7_df = pd.read_csv(ATLAS_TO_YEO7_CSV)
    
    # Define default groups if not provided
    if groups is None:
        groups = {
            'ICU': ['AF', 'DA2', 'PM', 'BA', 'VC'],
            'ICU Delirium': ['CG', 'DA', 'FS', 'FSE', 'GL', 'KJ', 'LL', 'MF', 'PM', 'PO', 'PB', 'SA'],
            'Healthy Controls': ['FEF', 'FD', 'GB', 'SG', 'AR', 'TL', 'TOG', 'PL', 'ZM', 'AM', 'PC', 'AD']
        }
    
    # Define region categories
    region_categories = [f'Yeo7_Network_{i}' for i in range(1, 8)] + ['Subcortical', 'Brainstem', 'Cerebellum']
    
    # Set default save path
    if save_path is None:
        results_dir = preproc_path / 'results'
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / 'sc_similarity_matrices.h5'
    
    # Check if file exists and should be loaded
    if save_path.exists() and not overwrite:
        print(f"Loading existing similarity matrices from {save_path}")
        if save_path.suffix == '.h5':
            return pd.read_hdf(save_path, key='similarity_matrices')
        else:
            return pd.read_csv(save_path)
    
    print("Computing similarity matrices between subjects for structural connectomes...")
    
    # Load all connectomes
    print(f"Loading {connectome_type} connectomes...")
    connectomes = load_all_connectomes(preproc_dir, connectome_type)
    print(f"Loaded {len(connectomes)} connectomes")
    
    results_rows = []
    
    # Process by group and region category
    for group_name, subject_list in groups.items():
        print(f"\nProcessing group: {group_name}")
        
        # Filter subjects that have connectomes
        subjects_with_connectomes = []
        for subj_id in subject_list:
            subj_full_id = f"sub-{subj_id}"
            if subj_full_id in connectomes:
                subjects_with_connectomes.append(subj_full_id)
        
        if len(subjects_with_connectomes) < 2:
            print(f"  Skipping {group_name}: need at least 2 subjects with connectomes")
            continue
        
        print(f"  Found {len(subjects_with_connectomes)} subjects with connectomes")
        
        # Extract region vectors for each subject
        region_vectors = {}
        for region_cat in region_categories:
            region_vectors[region_cat] = {}
            for subj in subjects_with_connectomes:
                try:
                    vec = extract_region_submatrix(connectomes[subj], region_cat, atlas_to_yeo7_df)
                    if len(vec) > 0:
                        region_vectors[region_cat][subj] = vec
                except Exception as e:
                    print(f"    Warning: Could not extract {region_cat} for {subj}: {e}")
                    continue
        
        # Compute pairwise similarities
        for region_cat in region_categories:
            if len(region_vectors[region_cat]) < 2:
                continue
            
            subjects_in_region = list(region_vectors[region_cat].keys())
            
            for subj_i, subj_j in combinations(subjects_in_region, 2):
                vec_i = region_vectors[region_cat][subj_i]
                vec_j = region_vectors[region_cat][subj_j]
                
                # Compute Pearson correlation
                if len(vec_i) == len(vec_j) and len(vec_i) > 0:
                    # Remove any NaN or Inf values
                    mask = np.isfinite(vec_i) & np.isfinite(vec_j)
                    if np.sum(mask) > 1:
                        similarity = np.corrcoef(vec_i[mask], vec_j[mask])[0, 1]
                        if np.isfinite(similarity):
                            results_rows.append({
                                'Group': group_name,
                                'Region Category': region_cat,
                                'Subject i': subj_i,
                                'Subject j': subj_j,
                                'Similarity': similarity
                            })
    
    similarity_df = pd.DataFrame(results_rows)
    
    # Save results
    if save_path.suffix == '.h5':
        similarity_df.to_hdf(save_path, key='similarity_matrices', mode='w')
    else:
        similarity_df.to_csv(save_path, index=False)
    
    print(f"\n✓ Saved similarity matrices to {save_path}")
    print(f"  Total pairs: {len(similarity_df)}")
    
    return similarity_df


def load_parcellation_mappings(
    atlas_to_yeo7_csv: Union[str, Path] = ATLAS_TO_YEO7_CSV,
    parcels_labels_xlsx: Union[str, Path] = PARCELS_LABELS_XLSX
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Load cortical Yeo7 mapping and subcortical category mapping.

    Returns
    -------
    parcel_to_yeo7 : dict
        Maps parcel_id (1-360) to Yeo7 network ID (1-7).
    parcel_to_category : dict
        Maps parcel_id (361-430, after removal of parcels 390 and 423) to subcortical category.
    """
    atlas_df = pd.read_csv(atlas_to_yeo7_csv)
    parcel_to_yeo7 = {
        int(row.parcel_id): int(row.yeo7_network)
        for _, row in atlas_df.iterrows()
        if pd.notna(row.yeo7_network) and int(row.parcel_id) <= 360 and int(row.yeo7_network) > 0
    }

    labels_df = pd.read_excel(parcels_labels_xlsx)
    parcel_to_category = {}

    basal_keywords = [
        'putamen', 'caudate', 'pallidus', 'accumbens',
        'ventral tegmental', 'septal', 'basalis', 'substantia nigra'
    ]

    for _, row in labels_df.iterrows():
        pid = int(row.parcel_index)
        if pid < 361 or pid > 430:
            continue

        desc = str(row.area_describtion).lower() if pd.notna(row.area_describtion) else ''
        name = str(row.area_name).lower() if pd.notna(row.area_name) else ''

        if 'cerebellum' in desc or 'cerebellum' in name or pid in (429, 430):
            category = 'Cerebellum'
        elif 'brainstem' in desc or pid in (425, 426, 427, 428):
            category = 'Brainstem'
        elif desc.startswith('thalamus') or 'thalamus' in desc:
            category = 'Thalamus'
        elif 'amyg' in desc or 'amyg' in name:
            category = 'Amygdala'
        elif any(k in desc for k in basal_keywords) or any(k in name for k in ['putam', 'caud', 'gpe', 'gpi', 'nac', 'septum', 'nb']):
            category = 'Basal Ganglia'
        else:
            category = 'Other'

        parcel_to_category[pid] = category

    return parcel_to_yeo7, parcel_to_category


def compute_overall_average_connectome(
    preproc_dir: Union[str, Path] = PREPROC_DIR,
    connectome_type: str = "SC_sift2_sizecorr"
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load all connectomes of a type and compute the overall average across subjects.

    Returns
    -------
    avg_connectome : np.ndarray
        Average 430×430 connectome.
    connectomes : dict
        Loaded subject connectomes (keyed by subject ID).
    """
    connectomes = load_all_connectomes(preproc_dir=preproc_dir, connectome_type=connectome_type)
    if len(connectomes) == 0:
        raise ValueError(f"No connectomes found for type {connectome_type}")

    avg_connectome = compute_group_average(connectomes)
    return avg_connectome, connectomes


def plot_overall_average_connectome(
    avg_connectome: np.ndarray,
    parcel_to_yeo7: Dict[int, int],
    parcel_to_category: Dict[int, str],
    yeo7_names: Optional[List[str]] = None,
    cmap: str = 'viridis',
    connectome_type: str = "SC_sift2_sizecorr",
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot the overall average connectome (all subjects) as a combined cortical+subcortical heatmap.
    """
    yeo7_names = yeo7_names or ['Visual', 'Somatomotor', 'DorsalAttn', 'VentralAttn', 'Limbic', 'FrontoParietal', 'Default']

    fig = plot_combined_connectome_cortical_subcortical(
        avg_connectome,
        parcel_to_yeo7,
        yeo7_names,
        parcel_to_category,
        title=f"All Subjects — Combined Connectome ({connectome_type})",
        cmap=cmap,
        save_path=save_path
    )
    return fig


def compute_full_subject_similarity_matrix(
    connectomes_dict: Dict[str, np.ndarray],
    expect_shape: Tuple[int, int] = (430, 430)
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute subject×subject Pearson correlation of full connectomes (vectorized upper triangle).
    """
    subjects = sorted(connectomes_dict.keys())
    if len(subjects) < 2:
        raise ValueError("Need at least two subjects to compute similarity.")

    triu_idx = np.triu_indices(expect_shape[0], k=1)
    vectors = []

    for sid in subjects:
        mat = prepare_matrix(connectomes_dict[sid])
        if mat.shape != expect_shape:
            raise ValueError(f"{sid} has shape {mat.shape}, expected {expect_shape}")
        vec = mat[triu_idx]
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vectors.append(vec)

    data = np.vstack(vectors)
    sim_matrix = np.corrcoef(data)
    sim_matrix = np.where(np.isfinite(sim_matrix), sim_matrix, np.nan)

    return sim_matrix, subjects


def plot_subject_similarity_heatmap(
    sim_matrix: np.ndarray,
    subjects: List[str],
    title: str = "Subject × Subject Similarity",
    cmap: str = "RdBu_r",
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot a heatmap of subject-by-subject similarity.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        ax=ax,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=subjects,
        yticklabels=subjects,
        cbar_kws={'label': 'Pearson r'}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='y', labelrotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def recover_sc_similarity_matrix(
    similarity_df: pd.DataFrame,
    group: str,
    region_category: str,
    subject_list: List[str]
) -> np.ndarray:
    """
    Recover symmetric similarity matrix from DataFrame for a specific group and region category.
    
    Parameters:
    -----------
    similarity_df : pd.DataFrame
        DataFrame with columns: 'Group', 'Region Category', 'Subject i', 'Subject j', 'Similarity'
    group : str
        Group name (e.g., 'ICU', 'ICU Delirium', 'Healthy Controls')
    region_category : str
        Region category name (e.g., 'Yeo7_Network_1', 'Subcortical')
    subject_list : list
        List of subject IDs (full IDs like 'sub-AF', 'sub-CG', etc.)
        Will be sorted to ensure consistent ordering
    
    Returns:
    --------
    matrix : np.ndarray
        N×N symmetric similarity matrix where N = len(subject_list)
        Diagonal is set to NaN
    """
    # Sort subject list to ensure consistent ordering
    subject_list_sorted = sorted(subject_list)
    
    # Filter DataFrame for this group and region category
    filtered_df = similarity_df[
        (similarity_df['Group'] == group) & 
        (similarity_df['Region Category'] == region_category)
    ]
    
    # Initialize matrix
    n_subjects = len(subject_list_sorted)
    matrix = np.zeros((n_subjects, n_subjects))
    
    # Fill matrix from DataFrame
    for _, row in filtered_df.iterrows():
        subj_i = row['Subject i']
        subj_j = row['Subject j']
        similarity = row['Similarity']
        
        if subj_i in subject_list_sorted and subj_j in subject_list_sorted:
            i_idx = subject_list_sorted.index(subj_i)
            j_idx = subject_list_sorted.index(subj_j)
            matrix[i_idx, j_idx] = similarity
            matrix[j_idx, i_idx] = similarity  # Symmetric
    
    # Set diagonal to NaN
    np.fill_diagonal(matrix, np.nan)
    
    # Validation: Check that we have data for all subject pairs
    expected_pairs = n_subjects * (n_subjects - 1) // 2
    actual_pairs = len(filtered_df)
    if actual_pairs < expected_pairs:
        warnings.warn(f"Warning: Only {actual_pairs} pairs found for {group}/{region_category}, expected {expected_pairs}")
    
    return matrix


def plot_sc_fingerprint_btw_subjects_by_group(
    similarity_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    region_categories: List[str],
    save_path: Path,
    connectome_type: str = "SC_sift2",
    scale: str = 'bygroup',
    add_sd: bool = True
):
    """
    Plot subject × subject similarity matrices for structural connectomes.
    
    Layout: Groups as rows, region categories as columns.
    Uses get_group_colors() for consistent coloring.
    Includes SD values in titles if add_sd=True.
    Title includes connectome metric (e.g., "SC_sift2").
    Uses 'coolwarm' colormap like functional version.
    
    Parameters:
    -----------
    similarity_df : pd.DataFrame
        DataFrame with similarity values (from compute_sc_similarity_between_subjects)
    groups : dict
        Dictionary mapping group names to subject ID lists
    region_categories : list
        List of region category names to plot
    save_path : Path
        Path to save the figure
    connectome_type : str
        Connectome type for title (default: "SC_sift2")
    scale : str
        Color scale option: 'bygroup', 'same', or 'auto' (default: 'bygroup')
    add_sd : bool
        If True, add standard deviation to subplot titles (default: True)
    """
    # Yeo7 network name mapping (define early)
    yeo7_network_names = {
        1: 'Visual',
        2: 'Somatomotor',
        3: 'Dorsal Attention',
        4: 'Ventral Attention',
        5: 'Limbic',
        6: 'Frontoparietal',
        7: 'Default Mode'
    }
    
    # Prepare region labels with actual Yeo7 network names
    region_labels = []
    for region_cat in region_categories:
        if region_cat.startswith('Yeo7_Network_'):
            # Extract network number from 'Yeo7_Network_1', 'Yeo7_Network_2', etc.
            network_num = int(region_cat.split('_')[-1])
            network_name = yeo7_network_names.get(network_num, f'Network {network_num}')
            region_labels.append(network_name)
        else:
            # For non-Yeo7 regions (Subcortical, Brainstem, Cerebellum), use as-is
            region_labels.append(region_cat)
    

    n_groups = len(groups)
    n_regions = len(region_categories)
    
    # Recover matrices for each group and region category
    similarity_matrices_bygroup = {}
    subject_lists_bygroup = {}  # Store sorted subject lists for labeling
    for group_name in groups.keys():
        similarity_matrices_bygroup[group_name] = {}
        # Get full subject IDs for this group and sort them
        subject_list = sorted([f"sub-{sid}" for sid in groups[group_name]])
        subject_lists_bygroup[group_name] = subject_list
        
        for region_cat in region_categories:
            matrix = recover_sc_similarity_matrix(similarity_df, group_name, region_cat, subject_list)
            # Validation: Check matrix dimensions match subject count
            if matrix.shape[0] != len(subject_list):
                raise ValueError(f"Matrix dimension mismatch for {group_name}/{region_cat}: "
                               f"matrix shape {matrix.shape[0]} != subject count {len(subject_list)}")
            # Warn if matrix is all zeros (no data for this region)
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0 and np.all(valid_values == 0):
                warnings.warn(f"No similarity data found for {group_name}/{region_cat}. "
                            f"Matrix will be plotted as all zeros. This may occur for regions "
                            f"with insufficient connectivity data (e.g., Cerebellum with only 2 parcels).")
            similarity_matrices_bygroup[group_name][region_cat] = matrix
    
    # Compute color scale limits
    if scale == 'bygroup':
        vmins = []
        vmaxs = []
        for group_name in groups.keys():
            all_values = []
            for region_cat in region_categories:
                matrix = similarity_matrices_bygroup[group_name][region_cat]
                valid_values = matrix[~np.isnan(matrix)]
                # Skip all-zero matrices (regions with no data)
                if len(valid_values) > 0 and np.any(valid_values != 0):
                    all_values.extend(valid_values)
            if len(all_values) > 0:
                vmin = np.percentile(all_values, 5)
                vmax = np.percentile(all_values, 95)
            else:
                vmin, vmax = -1, 1
            vmins.append(vmin)
            vmaxs.append(vmax)
    elif scale == 'same':
        all_values = []
        for group_name in groups.keys():
            for region_cat in region_categories:
                matrix = similarity_matrices_bygroup[group_name][region_cat]
                valid_values = matrix[~np.isnan(matrix)]
                # Skip all-zero matrices (regions with no data)
                if len(valid_values) > 0 and np.any(valid_values != 0):
                    all_values.extend(valid_values)
        if len(all_values) > 0:
            vmin = np.percentile(all_values, 5)
            vmax = np.percentile(all_values, 95)
        else:
            vmin, vmax = -1, 1
    
    fig, axes = plt.subplots(n_groups, n_regions, 
                            figsize=(4*n_regions, 4*n_groups),
                            gridspec_kw={'left': 0.1, 'right': 0.98, 
                                        'top': 0.93, 'bottom': 0.05,
                                        'wspace': 0.3, 'hspace': 0.35})
    
    # Handle single row or column case
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    if n_regions == 1:
        axes = axes.reshape(-1, 1)
    
    # Get group colors 
    group_colors = get_group_colors(groups.keys())
    
    # Plot matrices
    for j, group_name in enumerate(groups.keys()):
        for i, region_cat in enumerate(region_categories):
            ax = axes[j, i]
            matrix = similarity_matrices_bygroup[group_name][region_cat]
            
            if scale == 'same':
                im = ax.imshow(matrix, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks(np.linspace(vmin, vmax, 6))
            elif scale == 'bygroup':
                im = ax.imshow(matrix, cmap='coolwarm', vmin=vmins[j], vmax=vmaxs[j], aspect='equal', interpolation='nearest')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks(np.linspace(vmins[j], vmaxs[j], 6))
            else:  # auto
                im = ax.imshow(matrix, cmap='coolwarm', aspect='equal', interpolation='nearest')
                # For auto scale, get the actual data range from the matrix
                valid_data = matrix[~np.isnan(matrix)]
                if len(valid_data) > 0:
                    auto_vmin = np.percentile(valid_data, 5)
                    auto_vmax = np.percentile(valid_data, 95)
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_ticks(np.linspace(auto_vmin, auto_vmax, 6))
                else:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    # If no valid data, just use default ticks
                    cbar.set_ticks(6)  # This will use MaxNLocator with 6 ticks

            ax.grid(False)

            # Build title with region name and SD
            # Only show region name on top row (j == 0)
            if j == 0:  # Top row: show region name (network name)
                title_parts = [region_labels[i]]  # Use the mapped network name
                if add_sd:
                    std_val = np.nanstd(matrix)
                    title_parts.append(f'SD: {std_val:.3f}')
                ax.set_title(' | '.join(title_parts), fontsize=11, fontweight='bold', pad=10)
            elif add_sd:  # Other rows: only show SD if requested
                std_val = np.nanstd(matrix)
                ax.set_title(f'SD: {std_val:.3f}', fontsize=10, pad=10)
            
            # Set labels and tick marks
            n_subjects = matrix.shape[0]
            if j == n_groups - 1:  # Only label x-axis on bottom row
                ax.set_xlabel('Subjects', fontsize=10)
            if i == 0:  # Only label y-axis on left column
                ax.set_ylabel('Subjects', fontsize=10, fontweight='bold')

            ax.set_xticks([0, n_subjects-1])
            ax.set_yticks([0, n_subjects-1])
                        
            # Set tick positions to match matrix indices (0, 1, 2, ...)
            if n_subjects <= 10:
                tick_positions = list(range(n_subjects))
                tick_labels = [str(i) for i in range(n_subjects)]
            else:
                step = max(1, n_subjects // 5)
                tick_positions = list(range(0, n_subjects, step))
                if tick_positions[-1] != n_subjects - 1:
                    tick_positions.append(n_subjects - 1)
                tick_labels = [str(i) for i in tick_positions]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=8)
    
    # Add row labels (group names) on the left side using figure coordinates
    # This ensures proper alignment with rows regardless of subplot positions
    for j, group_name in enumerate(groups.keys()):
        # Get the leftmost axis in this row to determine position
        ax_ref = axes[j, 0]
        bbox = ax_ref.get_position()
        # Center y position of the row (in figure coordinates)
        y_center = bbox.y0 + bbox.height / 2
        # Place text to the left, using figure coordinates for proper alignment
        fig.text(0.06, y_center, group_name, transform=fig.transFigure,
                rotation=90, ha='center', va='center', fontsize=14, 
                fontweight='bold', color=group_colors[j])
    
    # Main title
    fig.suptitle(f'Subject × Subject Similarity Matrices (Pearson r) of Structural Connectivity ({connectome_type})',
                fontsize=16, fontweight='bold', y=0.98)
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f" Saved figure to {save_path}")


def compute_sc_similarity_all_subjects_by_region(
    connectomes_dict: Dict[str, np.ndarray],
    region_categories: List[str],
    atlas_to_yeo7_df: Optional[pd.DataFrame] = None
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Compute subject×subject similarity matrices for ALL subjects (not grouped) for each region category.
    
    Parameters:
    -----------
    connectomes_dict : dict
        Dictionary mapping subject IDs to connectome matrices
    region_categories : list
        List of region category names (e.g., ['Yeo7_Network_1', 'Yeo7_Network_2', ..., 'Subcortical'])
    atlas_to_yeo7_df : pd.DataFrame, optional
        DataFrame with columns 'parcel_id' and 'yeo7_network'. If None, loads from ATLAS_TO_YEO7_CSV.
    
    Returns:
    --------
    similarity_matrices : dict
        Dictionary mapping region_category to (similarity_matrix, subject_list) tuple
        similarity_matrix is N×N where N = number of subjects
        subject_list is sorted list of subject IDs
    """
    if atlas_to_yeo7_df is None:
        atlas_to_yeo7_df = pd.read_csv(ATLAS_TO_YEO7_CSV)
    
    # Get all subjects sorted
    all_subjects = sorted(connectomes_dict.keys())
    n_subjects = len(all_subjects)
    
    if n_subjects < 2:
        raise ValueError("Need at least 2 subjects to compute similarity")
    
    similarity_matrices = {}
    
    # Process each region category
    for region_cat in region_categories:
        print(f"Processing region: {region_cat}")
        
        # Extract region vectors for all subjects
        region_vectors = {}
        for subj in all_subjects:
            try:
                vec = extract_region_submatrix(connectomes_dict[subj], region_cat, atlas_to_yeo7_df)
                if len(vec) > 0:
                    region_vectors[subj] = vec
            except Exception as e:
                print(f"  Warning: Could not extract {region_cat} for {subj}: {e}")
                continue
        
        if len(region_vectors) < 2:
            print(f"  Skipping {region_cat}: need at least 2 subjects with valid data")
            continue
        
        # Get subjects that have valid vectors for this region
        subjects_with_data = sorted(region_vectors.keys())
        n_subj_region = len(subjects_with_data)
        
        # Initialize similarity matrix
        sim_matrix = np.eye(n_subj_region)  # Diagonal is 1 (self-similarity)
        
        # Compute pairwise similarities
        for i, subj_i in enumerate(subjects_with_data):
            for j, subj_j in enumerate(subjects_with_data):
                if i < j:
                    vec_i = region_vectors[subj_i]
                    vec_j = region_vectors[subj_j]
                    
                    # Compute Pearson correlation
                    if len(vec_i) == len(vec_j) and len(vec_i) > 0:
                        mask = np.isfinite(vec_i) & np.isfinite(vec_j)
                        if np.sum(mask) > 1:
                            correlation = np.corrcoef(vec_i[mask], vec_j[mask])[0, 1]
                            if np.isfinite(correlation):
                                sim_matrix[i, j] = correlation
                                sim_matrix[j, i] = correlation  # Symmetric
        
        similarity_matrices[region_cat] = (sim_matrix, subjects_with_data)
        print(f"  Computed similarity matrix for {n_subj_region} subjects")
    
    return similarity_matrices


def plot_sc_similarity_all_subjects_by_region(
    similarity_matrices: Dict[str, Tuple[np.ndarray, List[str]]],
    region_categories: List[str],
    connectome_type: str = "SC_sift2_sizecorr",
    scale: str = 'same',
    add_sd: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plot subject×subject similarity matrices for ALL subjects (not grouped) by region category.
    
    Layout: 1 row, region categories as columns. Regions with no data (all-zero or NaN matrices)
    are skipped from plotting and instead display a "No data" message.
    Similar to plot_sc_fingerprint_btw_subjects_by_group but for all subjects combined.
    
    Parameters:
    -----------
    similarity_matrices : dict
        Dictionary from compute_sc_similarity_all_subjects_by_region()
        Maps region_category to (similarity_matrix, subject_list) tuple
    region_categories : list
        List of region category names to plot (in order)
    connectome_type : str
        Connectome type for title
    scale : str
        Color scale option: 'same' or 'auto' (default: 'same')
    add_sd : bool
        If True, add standard deviation to subplot titles
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculates based on number of regions.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Yeo7 network name mapping
    yeo7_network_names = {
        1: 'Visual',
        2: 'Somatomotor',
        3: 'Dorsal Attention',
        4: 'Ventral Attention',
        5: 'Limbic',
        6: 'Frontoparietal',
        7: 'Default Mode'
    }
    
    # Prepare region labels with actual Yeo7 network names
    region_labels = []
    for region_cat in region_categories:
        if region_cat.startswith('Yeo7_Network_'):
            network_num = int(region_cat.split('_')[-1])
            network_name = yeo7_network_names.get(network_num, f'Network {network_num}')
            region_labels.append(network_name)
        else:
            region_labels.append(region_cat)
    
    n_regions = len(region_categories)
    
    # Compute color scale limits
    if scale == 'same':
        all_values = []
        for region_cat in region_categories:
            if region_cat in similarity_matrices:
                matrix, _ = similarity_matrices[region_cat]
                valid_values = matrix[~np.isnan(matrix)]
                if len(valid_values) > 0 and np.any(valid_values != 0):
                    all_values.extend(valid_values)
        if len(all_values) > 0:
            vmin = np.percentile(all_values, 5)
            vmax = np.percentile(all_values, 95)
        else:
            vmin, vmax = -1, 1
    
    # Set figure size
    if figsize is None:
        figsize = (4 * n_regions, 4)
    
    # Create figure with 1 row, n_regions columns
    fig, axes = plt.subplots(1, n_regions, 
                            figsize=figsize,
                            gridspec_kw={'left': 0.1, 'right': 0.98, 
                                        'top': 0.85, 'bottom': 0.15,
                                        'wspace': 0.3})
    
    # Handle single column case
    if n_regions == 1:
        axes = np.array([axes])
    
    # Plot matrices
    for i, region_cat in enumerate(region_categories):
        ax = axes[i]
        
        if region_cat not in similarity_matrices:
            ax.text(0.5, 0.5, f'No data\nfor {region_labels[i]}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(region_labels[i], fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')
            continue
        
        matrix, subjects = similarity_matrices[region_cat]
        n_subjects = len(subjects)
        
        # Skip matrices with no informative values (all zeros or all NaN)
        valid_data = matrix[~np.isnan(matrix)]
        if valid_data.size == 0 or np.all(valid_data == 0):
            ax.text(0.5, 0.5, f'No data\nfor {region_labels[i]}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(region_labels[i], fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')
            continue

        # Plot heatmap
        if scale == 'same':
            im = ax.imshow(matrix, cmap='coolwarm', vmin=vmin, vmax=vmax, 
                          aspect='equal', interpolation='nearest')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(np.linspace(vmin, vmax, 6))
        else:  # auto
            im = ax.imshow(matrix, cmap='coolwarm', aspect='equal', interpolation='nearest')
            valid_data = matrix[~np.isnan(matrix)]
            if len(valid_data) > 0:
                auto_vmin = np.percentile(valid_data, 5)
                auto_vmax = np.percentile(valid_data, 95)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks(np.linspace(auto_vmin, auto_vmax, 6))
            else:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks(6)
        
        ax.grid(False)
        
        # Build title with region name and SD
        title_parts = [region_labels[i]]
        if add_sd:
            std_val = np.nanstd(matrix)
            title_parts.append(f'SD: {std_val:.3f}')
        ax.set_title(' | '.join(title_parts), fontsize=11, fontweight='bold', pad=10)
        
        # Set labels
        ax.set_xlabel('Subjects', fontsize=10)
        if i == 0:
            ax.set_ylabel('Subjects', fontsize=10, fontweight='bold')
        
        # Set tick positions
        if n_subjects <= 10:
            tick_positions = list(range(n_subjects))
            tick_labels = [str(i) for i in range(n_subjects)]
        else:
            step = max(1, n_subjects // 5)
            tick_positions = list(range(0, n_subjects, step))
            if tick_positions[-1] != n_subjects - 1:
                tick_positions.append(n_subjects - 1)
            tick_labels = [str(i) for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)
    
    # Main title
    fig.suptitle(f'Subject × Subject Similarity Matrices (Pearson r) - All Subjects ({connectome_type})',
                fontsize=16, fontweight='bold', y=0.95)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    return fig


# ============================================================================
# CONNECTIVITY STRENGTH ANALYSIS FUNCTIONS
# ============================================================================

def compute_connectivity_strength_metrics(
    connectomes_dict: Dict[str, np.ndarray],
    groups: Dict[str, List[str]],
    region_type: str = 'whole_brain',
    parcel_to_yeo7: Optional[Dict[int, int]] = None,
    subcort_df: Optional[pd.DataFrame] = None,
    parcel_to_category: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Compute connectivity strength metrics for a given connectome type and region type.
    
    Parameters:
    -----------
    connectomes_dict : dict
        Dictionary of connectomes (from load_all_connectomes()), keys are subject IDs with 'sub-' prefix
    groups : dict
        Dictionary mapping group names to subject lists (without 'sub-' prefix)
    region_type : str
        'whole_brain' or 'subcortical'
    parcel_to_yeo7 : dict, optional
        Mapping dictionary from parcel_id (1-430) to Yeo7 network ID (for whole-brain only)
    subcort_df : pd.DataFrame, optional
        DataFrame with subcortical parcel info (for subcortical only)
    parcel_to_category : dict, optional
        Mapping from parcel_id to subcortical category (for subcortical only)
    
    Returns:
    --------
    metrics_df : pd.DataFrame
        DataFrame with columns: subject, group, total_strength, avg_node_strength, 
        [network-specific or category-specific columns]
    """
    connectivity_metrics = []
    
    for group_name, group_subjects in groups.items():
        for subj in group_subjects:
            # Check if subject exists in connectomes_dict (with 'sub-' prefix)
            subj_key = f'sub-{subj}' if not subj.startswith('sub-') else subj
            if subj_key not in connectomes_dict:
                continue
            
            # Prepare matrix
            matrix = prepare_matrix(connectomes_dict[subj_key])
            
            if region_type == 'whole_brain':
                # Whole-brain analysis
                # 1. Total connectivity strength
                total_strength = np.sum(matrix)
                
                # 2. Average node strength
                node_strengths = np.sum(matrix, axis=1)
                avg_node_strength = np.mean(node_strengths)
                
                # 3. Strength per Yeo7 network
                if parcel_to_yeo7 is None:
                    raise ValueError("parcel_to_yeo7 must be provided for whole_brain analysis")
                
                yeo7_strengths = {}
                for parcel_id in range(1, 431):  # 1-430
                    if parcel_id in parcel_to_yeo7:
                        yeo7_net = parcel_to_yeo7[parcel_id]
                        node_idx = parcel_id - 1  # Convert to 0-indexed
                        
                        if yeo7_net not in yeo7_strengths:
                            yeo7_strengths[yeo7_net] = []
                        
                        yeo7_strengths[yeo7_net].append(node_strengths[node_idx])
                
                # Compute mean strength per network
                yeo7_mean_strengths = {}
                for net_id, strengths in yeo7_strengths.items():
                    if len(strengths) > 0:
                        yeo7_mean_strengths[f'yeo7_net_{net_id}_strength'] = np.mean(strengths)
                
                # Store metrics
                metrics = {
                    'subject': subj,
                    'group': group_name,
                    'total_strength': total_strength,
                    'avg_node_strength': avg_node_strength,
                    **yeo7_mean_strengths
                }
                
            elif region_type == 'subcortical':
                # Subcortical analysis
                # Extract subcortical submatrix (parcels 361-424, indices 360-423)
                subcort_indices = list(range(360, 424))  # 0-indexed: 360-423
                subcort_matrix = matrix[np.ix_(subcort_indices, subcort_indices)]
                
                # 1. Total subcortical connectivity strength
                total_subcort_strength = np.sum(subcort_matrix)
                
                # 2. Average subcortical node strength
                subcort_node_strengths = np.sum(subcort_matrix, axis=1)
                avg_subcort_node_strength = np.mean(subcort_node_strengths)
                
                # 3. Strength per category
                if parcel_to_category is None:
                    raise ValueError("parcel_to_category must be provided for subcortical analysis")
                
                category_strengths = {}
                for category in ['Thalamus', 'Basal Ganglia', 'Brainstem', 'Amygdala', 'Cerebellum']:
                    # Find parcels in this category
                    category_parcels = [pid for pid, cat in parcel_to_category.items() if cat == category]
                    if len(category_parcels) > 0:
                        # Get indices for these parcels within subcortical matrix
                        category_indices = [pid - 1 - 360 for pid in category_parcels if 361 <= pid <= 424]
                        category_indices = [idx for idx in category_indices if 0 <= idx < 64]
                        
                        if len(category_indices) > 0:
                            # Get node strengths for these nodes from the full subcortical matrix
                            cat_node_strengths = subcort_node_strengths[category_indices]
                            category_strengths[category] = np.mean(cat_node_strengths)
                
                # Store metrics
                metrics = {
                    'subject': subj,
                    'group': group_name,
                    'total_subcort_strength': total_subcort_strength,
                    'avg_subcort_node_strength': avg_subcort_node_strength,
                    **{f'category_{cat.replace(" ", "_").lower()}_strength': val 
                       for cat, val in category_strengths.items()}
                }
            else:
                raise ValueError(f"Unknown region_type: {region_type}. Must be 'whole_brain' or 'subcortical'")
            
            connectivity_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(connectivity_metrics)
    return metrics_df


def _run_statistical_tests_base(
    metrics_df: pd.DataFrame,
    metric_col: str,
    groups: Dict[str, List[str]],
    title: str,
    verbose: bool = True,
    value_format: str = '.4e'
) -> Dict:
    """
    Base function for running comprehensive statistical tests (assumption checks, ANOVA, Kruskal-Wallis, post-hoc).
    This function contains the common logic shared by run_connectivity_statistical_tests and run_efficiency_statistical_tests.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with metrics
    metric_col : str
        Column name of the metric to test
    groups : dict
        Dictionary of groups
    title : str
        Title for the statistical test section
    verbose : bool
        Whether to print results
    value_format : str
        Format string for printing values (default: '.4e' for scientific notation)
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    from scipy import stats
    from scipy.stats import shapiro, levene, f_oneway, ttest_ind, kruskal, mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    
    group_names = list(groups.keys())
    all_groups_data = [metrics_df[metrics_df['group'] == g][metric_col].dropna().values 
                       for g in group_names]
    
    # Filter out groups with no data
    valid_groups = [(name, data) for name, data in zip(group_names, all_groups_data) if len(data) > 0]
    if len(valid_groups) < 2:
        if verbose:
            print(f"Not enough groups with data for {metric_col}")
        return {}
    
    group_names = [name for name, _ in valid_groups]
    all_groups_data = [data for _, data in valid_groups]
    
    results = {
        'metric_name': metric_col,
        'group_names': group_names,
        'group_data': {name: data.tolist() for name, data in zip(group_names, all_groups_data)}
    }
    
    if verbose:
        print(f"\n{title}")
        print("-" * 80)
        for group_name, data in zip(group_names, all_groups_data):
            print(f"{group_name}:")
            print(f"  Mean: {np.mean(data):{value_format}} ± {np.std(data):{value_format}}")
            print(f"  Median: {np.median(data):{value_format}}")
            print(f"  Range: [{np.min(data):{value_format}}, {np.max(data):{value_format}}]")
            print(f"  N: {len(data)}")
    
    # Check assumptions for parametric ANOVA
    if verbose:
        print("\nAssumption checks for parametric ANOVA:")
    
    normality_pvals = []
    for i, (group_name, data) in enumerate(zip(group_names, all_groups_data)):
        if len(data) >= 3 and len(data) <= 5000:
            try:
                _, p_norm = shapiro(data)
                normality_pvals.append(p_norm)
                if verbose:
                    print(f"  {group_name}: Shapiro-Wilk p={p_norm:.6f} {'(normal)' if p_norm > 0.05 else '(non-normal)'}")
            except:
                if verbose:
                    print(f"  {group_name}: Cannot test normality (n={len(data)})")
        else:
            if verbose:
                print(f"  {group_name}: Sample size {len(data)} outside Shapiro-Wilk range")
    
    # Equal variances test (Levene's test)
    try:
        levene_stat, p_levene = levene(*all_groups_data)
        results['levene'] = {'statistic': levene_stat, 'pvalue': p_levene}
        if verbose:
            print(f"  Levene's test (equal variances): p={p_levene:.6f} {'(equal variances)' if p_levene > 0.05 else '(unequal variances)'}")
    except:
        if verbose:
            print(f"  Cannot perform Levene's test")
        results['levene'] = None
    
    # Decide: if assumptions met, use ANOVA; otherwise use Kruskal-Wallis
    assumptions_met = all(p > 0.05 for p in normality_pvals) if len(normality_pvals) == len(all_groups_data) else False
    assumptions_met = assumptions_met and (p_levene > 0.05 if results.get('levene') and results['levene'] else False)
    results['assumptions_met'] = assumptions_met
    
    if verbose:
        print(f"\n  → Assumptions for ANOVA: {'MET' if assumptions_met else 'NOT MET'}")
        print(f"  → Recommendation: Use {'parametric ANOVA' if assumptions_met else 'non-parametric Kruskal-Wallis'}")
    
    # Option 1: One-way ANOVA (parametric)
    if verbose:
        print("\n1. ONE-WAY ANOVA (Parametric):")
        print("-" * 80)
    
    try:
        f_stat, pval_anova = f_oneway(*all_groups_data)
        results['anova'] = {'f_statistic': f_stat, 'pvalue': pval_anova}
        if verbose:
            print(f"  F-statistic: {f_stat:.4f}, p-value: {pval_anova:.6f}")
            print(f"  {'✓ Significant differences detected' if pval_anova < 0.05 else '✗ No significant differences'} (α=0.05)")
        
        if pval_anova < 0.05:
            # Post-hoc: pairwise t-tests with FDR correction
            if verbose:
                print("\n  Post-hoc tests (Welch's t-test with FDR correction):")
            
            pairwise_pvals = []
            pairwise_comparisons = []
            t_stats = []
            
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    g1_data = all_groups_data[i]
                    g2_data = all_groups_data[j]
                    
                    t_stat, pval = ttest_ind(g1_data, g2_data, equal_var=False)
                    pairwise_pvals.append(pval)
                    pairwise_comparisons.append((group_names[i], group_names[j]))
                    t_stats.append(t_stat)
            
            if len(pairwise_pvals) > 0:
                _, pvals_corrected, _, _ = multipletests(pairwise_pvals, method='fdr_bh', alpha=0.05)
                results['anova_posthoc'] = [
                    {'group1': g1, 'group2': g2, 't_statistic': t_stat, 'pvalue_uncorrected': pval_uncorr, 
                     'pvalue_corrected': pval_corr}
                    for (g1, g2), t_stat, pval_uncorr, pval_corr in zip(pairwise_comparisons, t_stats, pairwise_pvals, pvals_corrected)
                ]
                
                if verbose:
                    for (g1, g2), t_stat, pval_uncorr, pval_corr in zip(pairwise_comparisons, t_stats, pairwise_pvals, pvals_corrected):
                        sig = '✓' if pval_corr < 0.05 else '✗'
                        print(f"    {sig} {g1} vs {g2}: t={t_stat:.4f}, p={pval_uncorr:.6f} (FDR-corrected: {pval_corr:.6f})")
    except Exception as e:
        if verbose:
            print(f"  Cannot perform ANOVA: {e}")
        results['anova'] = None
    
    # Option 2: Kruskal-Wallis test (non-parametric) - RECOMMENDED
    if verbose:
        print("\n2. KRUSKAL-WALLIS TEST (Non-parametric) - RECOMMENDED:")
        print("-" * 80)
    
    try:
        h_stat, pval_kw = kruskal(*all_groups_data)
        results['kruskal_wallis'] = {'h_statistic': h_stat, 'pvalue': pval_kw}
        if verbose:
            print(f"  H-statistic: {h_stat:.4f}, p-value: {pval_kw:.6f}")
            print(f"  {'✓ Significant differences detected' if pval_kw < 0.05 else '✗ No significant differences'} (α=0.05)")
        
        # Post-hoc: Pairwise Mann-Whitney U tests with FDR correction
        if pval_kw < 0.05:
            if verbose:
                print("\n  Post-hoc pairwise comparisons (Mann-Whitney U test with FDR correction):")
            
            pairwise_pvals = []
            pairwise_comparisons = []
            u_stats = []
            
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    g1_data = all_groups_data[i]
                    g2_data = all_groups_data[j]
                    
                    try:
                        u_stat, pval = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                        pairwise_pvals.append(pval)
                        pairwise_comparisons.append((group_names[i], group_names[j]))
                        u_stats.append(u_stat)
                    except ValueError:
                        pass
            
            if len(pairwise_pvals) > 0:
                _, pvals_corrected, _, _ = multipletests(pairwise_pvals, method='fdr_bh', alpha=0.05)
                results['kruskal_wallis_posthoc'] = [
                    {'group1': g1, 'group2': g2, 'u_statistic': u_stat, 'pvalue_uncorrected': pval_uncorr, 
                     'pvalue_corrected': pval_corr}
                    for (g1, g2), u_stat, pval_uncorr, pval_corr in zip(pairwise_comparisons, u_stats, pairwise_pvals, pvals_corrected)
                ]
                
                if verbose:
                    for (g1, g2), u_stat, pval_uncorr, pval_corr in zip(pairwise_comparisons, u_stats, pairwise_pvals, pvals_corrected):
                        sig = '✓' if pval_corr < 0.05 else '✗'
                        print(f"    {sig} {g1} vs {g2}: U={u_stat:.2f}, p={pval_uncorr:.6f} (FDR-corrected: {pval_corr:.6f})")
        else:
            if verbose:
                print("\n  Overall test not significant - skipping pairwise comparisons")
            results['kruskal_wallis_posthoc'] = []
    except ValueError as e:
        if verbose:
            print(f"  Cannot perform Kruskal-Wallis test: {e}")
        results['kruskal_wallis'] = None
    
    # Summary recommendation
    if verbose:
        print("\n" + "="*80)
        print("STATISTICAL RECOMMENDATION:")
        print("="*80)
        if assumptions_met:
            print("  ✓ Assumptions met → Both ANOVA and Kruskal-Wallis are valid")
            print("  → Report both, but prioritize ANOVA (more powerful when assumptions met)")
        else:
            print("  ✗ Assumptions NOT met → Use Kruskal-Wallis (non-parametric)")
            print("  → ANOVA results should be interpreted with caution")
        print("="*80)
    
    results['recommendation'] = 'ANOVA' if assumptions_met else 'Kruskal-Wallis'
    
    return results


def run_connectivity_statistical_tests(
    metrics_df: pd.DataFrame,
    metric_name: str,
    groups: Dict[str, List[str]],
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive statistical tests (assumption checks, ANOVA, Kruskal-Wallis, post-hoc).
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame from compute_connectivity_strength_metrics()
    metric_name : str
        Name of metric to test (e.g., 'total_strength', 'avg_node_strength', 'yeo7_net_1_strength')
    groups : dict
        Dictionary of groups
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    # Use base function with connectivity-specific parameters
    title = metric_name.upper().replace('_', ' ')
    return _run_statistical_tests_base(
        metrics_df=metrics_df,
        metric_col=metric_name,
        groups=groups,
        title=title,
        verbose=verbose,
        value_format='.4e'  # Scientific notation for connectivity strength values
    )


def plot_connectivity_strength_results(
    metrics_df: pd.DataFrame,
    connectome_type: str,
    region_type: str = 'whole_brain',
    groups: Optional[Dict[str, List[str]]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create visualization plots (boxplots, bar charts) for connectivity strength results.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with metrics from compute_connectivity_strength_metrics()
    connectome_type : str
        Connectome type (e.g., 'SC_sift2_sizecorr', 'MEAN_FA', 'MEAN_MD')
    region_type : str
        'whole_brain' or 'subcortical'
    groups : dict, optional
        Dictionary of groups. If None, uses unique groups from metrics_df
    save_path : str or Path, optional
        Path to save figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if groups is None:
        # Extract groups from metrics_df
        unique_groups = metrics_df['group'].unique().tolist()
        groups = {g: [] for g in unique_groups}
    
    # Define group order and colors
    group_order = ['Healthy Controls', 'ICU', 'ICU Delirium']
    # Filter to only groups present in data
    group_order = [g for g in group_order if g in groups or g in metrics_df['group'].unique()]
    colors = get_group_colors(group_order)
    
    if region_type == 'whole_brain':
        # Whole-brain visualization: 2 plots (avg node strength, strength per network)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average node strength by group
        ax1 = axes[0]
        data_to_plot = [metrics_df[metrics_df['group'] == g]['avg_node_strength'].values for g in group_order]
        bp1 = ax1.boxplot(data_to_plot, labels=group_order, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Average Node Strength', fontsize=12, fontweight='bold')
        ax1.set_title(f'{connectome_type}: Average Node Strength by Group', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Strength per Yeo7 network
        ax2 = axes[1]
        yeo7_cols = [col for col in metrics_df.columns if col.startswith('yeo7_net_')]
        yeo7_net_ids = sorted([int(col.split('_')[2]) for col in yeo7_cols])
        
        # Yeo7 network names
        yeo7_names = {
            0: 'Subcortical/Unassigned',
            1: 'Visual',
            2: 'Somatomotor',
            3: 'Dorsal Attention',
            4: 'Ventral Attention',
            5: 'Limbic',
            6: 'Frontoparietal',
            7: 'Default Mode'
        }
        
        cortical_nets = [n for n in yeo7_net_ids if n > 0]
        if len(cortical_nets) > 0:
            net_data = []
            net_labels = []
            for net_id in cortical_nets[:7]:  # Limit to first 7 networks
                col_name = f'yeo7_net_{net_id}_strength'
                if col_name in metrics_df.columns:
                    net_labels.append(yeo7_names.get(net_id, f'Net{net_id}'))
                    group_means = []
                    for group_name in group_order:
                        group_data = metrics_df[metrics_df['group'] == group_name][col_name].dropna()
                        group_means.append(group_data.mean() if len(group_data) > 0 else 0)
                    net_data.append(group_means)
            
            if len(net_data) > 0:
                net_data = np.array(net_data)
                x = np.arange(len(net_labels))
                width = 0.25
                for i, group_name in enumerate(group_order):
                    ax2.bar(
                        x + i*width,
                        net_data[:, i],
                        width,
                        label=group_name,
                        color=colors[i],
                        alpha=0.7
                    )
                ax2.set_xlabel('Yeo7 Network', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Mean Network Strength', fontsize=12, fontweight='bold')
                ax2.set_title(f'{connectome_type}: Strength per Yeo7 Network', fontsize=13, fontweight='bold')
                ax2.set_xticks(x + width)
                ax2.set_xticklabels(net_labels, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
    elif region_type == 'subcortical':
        # Subcortical visualization: 2 plots (avg node strength, strength per category)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average subcortical node strength by group
        ax1 = axes[0]
        data_to_plot = [metrics_df[metrics_df['group'] == g]['avg_subcort_node_strength'].values for g in group_order]
        bp1 = ax1.boxplot(data_to_plot, labels=group_order, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Average Subcortical Node Strength', fontsize=12, fontweight='bold')
        ax1.set_title(f'{connectome_type}: Average Subcortical Node Strength by Group', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Strength per subcortical category
        ax2 = axes[1]
        category_cols = [col for col in metrics_df.columns if col.startswith('category_') and col.endswith('_strength')]
        categories = [col.replace('category_', '').replace('_strength', '').replace('_', ' ').title() 
                     for col in category_cols]
        
        if len(categories) > 0:
            cat_data = []
            cat_labels = []
            for col in category_cols:
                cat_labels.append(col.replace('category_', '').replace('_strength', '').replace('_', ' ').title())
                group_means = []
                for group_name in group_order:
                    group_data = metrics_df[metrics_df['group'] == group_name][col].dropna()
                    group_means.append(group_data.mean() if len(group_data) > 0 else 0)
                cat_data.append(group_means)
            
            if len(cat_data) > 0:
                cat_data = np.array(cat_data)
                x = np.arange(len(cat_labels))
                width = 0.25
                for i, group_name in enumerate(group_order):
                    ax2.bar(
                        x + i*width,
                        cat_data[:, i],
                        width,
                        label=group_name,
                        color=colors[i],
                        alpha=0.7
                    )
                ax2.set_xlabel('Subcortical Category', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Mean Category Strength', fontsize=12, fontweight='bold')
                ax2.set_title(f'{connectome_type}: Strength per Subcortical Category', fontsize=13, fontweight='bold')
                ax2.set_xticks(x + width)
                ax2.set_xticklabels(cat_labels, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
    else:
        raise ValueError(f"Unknown region_type: {region_type}. Must be 'whole_brain' or 'subcortical'")
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def run_full_connectivity_analysis(
    preproc_dir: Union[str, Path],
    connectome_type: str,
    groups: Dict[str, List[str]],
    region_type: str = 'whole_brain',
    save_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    parcel_to_yeo7: Optional[Dict[int, int]] = None,
    subcort_df: Optional[pd.DataFrame] = None,
    parcel_to_category: Optional[Dict[int, str]] = None
) -> Dict:
    """
    High-level function that runs the complete connectivity strength analysis pipeline.
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Preprocessing directory path
    connectome_type : str
        Connectome type to analyze (e.g., 'SC_sift2_sizecorr', 'MEAN_FA', 'MEAN_MD')
    groups : dict
        Dictionary of groups mapping group names to subject lists
    region_type : str
        'whole_brain' or 'subcortical'
    save_dir : str or Path, optional
        Directory to save results. If None, saves to preproc_dir/results/
    verbose : bool
        Whether to print progress
    parcel_to_yeo7 : dict, optional
        Mapping from parcel_id to Yeo7 network ID (required for whole_brain)
    subcort_df : pd.DataFrame, optional
        DataFrame with subcortical parcel info (not used directly, but kept for compatibility)
    parcel_to_category : dict, optional
        Mapping from parcel_id to subcortical category (required for subcortical)
    
    Returns:
    --------
    results : dict
        Dictionary with keys:
        - 'metrics_df': DataFrame with computed metrics
        - 'statistical_results': Dictionary of statistical test results
        - 'figure': matplotlib Figure object
        - 'save_paths': Dictionary of saved file paths
    """
    preproc_dir = Path(preproc_dir)
    if save_dir is None:
        save_dir = preproc_dir / 'results'
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*80)
        print(f"CONNECTIVITY STRENGTH ANALYSIS: {connectome_type} ({region_type})")
        print("="*80)
    
    # Load connectomes
    if verbose:
        print(f"\nLoading connectomes of type: {connectome_type}")
    connectomes_dict = load_all_connectomes(preproc_dir, connectome_type)
    
    if len(connectomes_dict) == 0:
        raise ValueError(f"No connectomes found for type: {connectome_type}")
    
    if verbose:
        print(f"Loaded {len(connectomes_dict)} connectomes")
    
    # Compute metrics
    if verbose:
        print(f"\nComputing connectivity strength metrics ({region_type})...")
    
    metrics_df = compute_connectivity_strength_metrics(
        connectomes_dict=connectomes_dict,
        groups=groups,
        region_type=region_type,
        parcel_to_yeo7=parcel_to_yeo7,
        subcort_df=subcort_df,
        parcel_to_category=parcel_to_category
    )
    
    if verbose:
        print(f"Computed metrics for {len(metrics_df)} subjects")
        print(f"Groups: {metrics_df['group'].value_counts().to_dict()}")
    
    # Run statistical tests
    if verbose:
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
    
    statistical_results = {}
    
    if region_type == 'whole_brain':
        # Test total_strength, avg_node_strength, and each network
        test_metrics = ['total_strength', 'avg_node_strength']
        yeo7_cols = [col for col in metrics_df.columns if col.startswith('yeo7_net_')]
        test_metrics.extend(yeo7_cols)
    else:  # subcortical
        # Test total_subcort_strength, avg_subcort_node_strength, and each category
        test_metrics = ['total_subcort_strength', 'avg_subcort_node_strength']
        category_cols = [col for col in metrics_df.columns if col.startswith('category_') and col.endswith('_strength')]
        test_metrics.extend(category_cols)
    
    for metric_name in test_metrics:
        if metric_name in metrics_df.columns:
            if verbose:
                print(f"\n{'='*80}")
            results = run_connectivity_statistical_tests(
                metrics_df=metrics_df,
                metric_name=metric_name,
                groups=groups,
                verbose=verbose
            )
            statistical_results[metric_name] = results
    
    # Create visualization
    if verbose:
        print("\n" + "="*80)
        print("VISUALIZATION")
        print("="*80)
    
    fig_save_path = save_dir / f'connectivity_strength_{region_type}_{connectome_type}.png'
    fig = plot_connectivity_strength_results(
        metrics_df=metrics_df,
        connectome_type=connectome_type,
        region_type=region_type,
        groups=groups,
        save_path=fig_save_path
    )
    
    # Save metrics DataFrame
    metrics_save_path = save_dir / f'connectivity_strength_{region_type}_{connectome_type}.csv'
    metrics_df.to_csv(metrics_save_path, index=False)
    if verbose:
        print(f"Metrics saved to {metrics_save_path}")
    
    return {
        'metrics_df': metrics_df,
        'statistical_results': statistical_results,
        'figure': fig,
        'save_paths': {
            'figure': fig_save_path,
            'metrics': metrics_save_path
        }
    }


# ============================================================================
# EFFICIENCY / PATH LENGTH ANALYSIS FUNCTIONS
# ============================================================================

def compute_efficiency_pathlength(matrix, use_bct=False, epsilon=1e-6):
    """
    Compute global efficiency and characteristic path length from a weighted
    connectivity matrix. Works with either bctpy or NetworkX.

    CRITICAL DESIGN DECISION:
    - Zeros in the matrix are treated as NO EDGE (not as edges with huge distance)
    - This preserves topology differences between groups
    - Only edges with W > 0 are included in the graph
    - Distance = 1/W for existing edges (stronger connection = shorter distance)

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric, non-negative weight matrix (e.g. SIFT2).
        Zeros indicate absence of connection (no edge).
    use_bct : bool
        If True, use bctpy; otherwise use NetworkX.
    epsilon : float
        Small value to avoid division by zero when computing 1/W for existing edges.

    Returns
    -------
    efficiency : float
        Global efficiency (mean of 1/distance over all pairs in largest connected component)
    char_path_length : float
        Characteristic path length (mean distance over all pairs in largest connected component)
    """
    # Try to import bctpy if requested, fall back to NetworkX if not available
    bct_available = False
    if use_bct:
        try:
            import bctpy as bct
            bct_available = True
        except ImportError:
            # Fall back to NetworkX if bctpy is not available
            import networkx as nx
            bct_available = False
    else:
        import networkx as nx
    
    W = matrix.astype(float).copy()
    np.fill_diagonal(W, 0.0)

    # Remove isolated nodes / keep largest connected component
    node_strength = W.sum(axis=0) + W.sum(axis=1)
    keep = np.where(node_strength > 0)[0]
    if keep.size < 2:
        return np.nan, np.nan

    W = W[np.ix_(keep, keep)]

    if use_bct and bct_available:
        # BCT branch: strengths -> distance_wei -> charpath
        # Note: bct.distance_wei correctly treats zeros as no edge (only considers W > 0)
        D, _ = bct.distance_wei(W)
        lambda_, efficiency, ecc, radius, diameter = bct.charpath(
            D,
            include_diagonal=False,
            include_infinite=False   # ignore disconnected pairs
        )
        char_path_length = lambda_

    else:
        # NetworkX branch: explicit distances + connected component
        # CRITICAL: Only create edges where W > 0 (treat zeros as NO EDGE, not huge edges)
        # This preserves topology differences between groups
        G = nx.Graph()
        rows, cols = np.where(W > 0)
        for i, j in zip(rows, cols):
            if i < j:  # undirected graph, only add each edge once
                # Distance is inverse of weight: stronger connection = shorter distance
                length_ij = 1.0 / (W[i, j] + epsilon)
                G.add_edge(i, j, weight=length_ij)

        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return np.nan, np.nan

        # Work on largest connected component only
        components = list(nx.connected_components(G))
        if not components:
            return np.nan, np.nan
        largest_cc = max(components, key=len)
        Gc = G.subgraph(largest_cc).copy()

        if Gc.number_of_nodes() < 2:
            return np.nan, np.nan

        # Compute all pairs shortest paths
        lengths_dict = dict(nx.all_pairs_dijkstra_path_length(Gc, weight="weight"))
        nodes = list(Gc.nodes())

        d_vals = []
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j:  # only count each pair once
                    if v in lengths_dict[u]:
                        d_vals.append(lengths_dict[u][v])

        if len(d_vals) == 0:
            return np.nan, np.nan

        d_vals = np.array(d_vals)
        char_path_length = d_vals.mean()
        efficiency = (1.0 / d_vals).mean()

    return efficiency, char_path_length


def compute_efficiency_pathlength_metrics(
    connectomes_dict: Dict[str, np.ndarray],
    groups: Dict[str, List[str]],
    region_type: str = 'whole_brain',
    target_density: Optional[float] = None,
    use_bct: bool = False,
    parcel_to_yeo7: Optional[Dict[int, int]] = None,
    parcel_to_category: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Compute efficiency and path length metrics for a given connectome type and region type.
    Uses global normalization + fixed threshold approach for fair topology comparison.
    
    Parameters:
    -----------
    connectomes_dict : dict
        Dictionary of connectomes (from load_all_connectomes()), keys are subject IDs with 'sub-' prefix
    groups : dict
        Dictionary mapping group names to subject lists (without 'sub-' prefix)
    region_type : str
        'whole_brain' or 'subcortical'
    target_density : float, optional
        Target density for thresholding (e.g., 0.15 for top 15% of edges). If None, no thresholding.
    use_bct : bool
        If True, use bctpy; otherwise use NetworkX
    parcel_to_yeo7 : dict, optional
        Mapping dictionary from parcel_id (1-430) to Yeo7 network ID (for whole-brain only)
    parcel_to_category : dict, optional
        Mapping from parcel_id to subcortical category (for subcortical only)
    
    Returns:
    --------
    metrics_df : pd.DataFrame
        DataFrame with columns: subject, group, global_efficiency, char_path_length,
        [network-specific or category-specific columns]
    """
    # STEP 1: Collect all matrices and compute global normalization factor
    all_matrices = []
    all_subjects = []
    
    for group_name, group_subjects in groups.items():
        for subj in group_subjects:
            subj_key = f'sub-{subj}' if not subj.startswith('sub-') else subj
            if subj_key not in connectomes_dict:
                continue
            
            matrix = prepare_matrix(connectomes_dict[subj_key])
            all_matrices.append((subj, group_name, matrix))
            all_subjects.append((subj, group_name))
    
    if len(all_matrices) == 0:
        return pd.DataFrame()
    
    # Compute global normalization factor (mean of all non-zero weights across all subjects)
    all_nonzero_weights = np.concatenate([m[m > 0] for _, _, m in all_matrices])
    if len(all_nonzero_weights) == 0:
        return pd.DataFrame()
    
    global_mean_weight = all_nonzero_weights.mean()
    
    # Normalize all matrices by global mean
    all_matrices_normalized = []
    for subj, group_name, matrix in all_matrices:
        matrix_normalized = matrix.copy()
        if global_mean_weight > 0:
            matrix_normalized = matrix_normalized / global_mean_weight
        all_matrices_normalized.append((subj, group_name, matrix_normalized))
    
    # Compute global threshold if density thresholding is requested
    if target_density and target_density < 1.0:
        all_normalized_weights = np.concatenate([m[m > 0] for _, _, m in all_matrices_normalized])
        if len(all_normalized_weights) > 0:
            global_threshold = np.percentile(all_normalized_weights, (1 - target_density) * 100)
        else:
            global_threshold = None
    else:
        global_threshold = None
    
    # STEP 2: Process each subject with normalized weights and thresholding
    efficiency_metrics = []
    
    for (subj, group_name), (_, _, matrix_normalized) in zip(all_subjects, all_matrices_normalized):
        # Apply global threshold if specified
        if global_threshold is not None:
            matrix_thresh = matrix_normalized.copy()
            matrix_thresh[matrix_thresh < global_threshold] = 0.0
        else:
            matrix_thresh = matrix_normalized.copy()
        
        # Ensure symmetric and zero diagonal
        matrix_thresh = 0.5 * (matrix_thresh + matrix_thresh.T)
        np.fill_diagonal(matrix_thresh, 0.0)
        
        if region_type == 'whole_brain':
            # Whole-brain analysis
            # 1. Global efficiency and path length
            efficiency, char_path = compute_efficiency_pathlength(
                matrix_thresh,
                use_bct=use_bct
            )
            
            metrics = {
                'subject': subj,
                'group': group_name,
                'global_efficiency': efficiency,
                'char_path_length': char_path
            }
            
            # 2. Network-specific efficiency and path length
            if parcel_to_yeo7 is not None:
                for net_id in range(8):  # Networks 0-7
                    # Extract submatrix for this network
                    net_indices = [i for i in range(430) if parcel_to_yeo7.get(i+1, 0) == net_id]
                    if len(net_indices) < 2:
                        continue
                    
                    submatrix = matrix_thresh[np.ix_(net_indices, net_indices)]
                    net_efficiency, net_path = compute_efficiency_pathlength(
                        submatrix,
                        use_bct=use_bct
                    )
                    metrics[f'yeo7_net_{net_id}_efficiency'] = net_efficiency
                    metrics[f'yeo7_net_{net_id}_pathlength'] = net_path
            
            efficiency_metrics.append(metrics)
            
        elif region_type == 'subcortical':
            # Subcortical analysis
            # Extract subcortical submatrix (parcels 361-424, indices 360-423)
            subcort_indices = list(range(360, 424))  # 0-indexed: 360-423
            subcort_matrix = matrix_thresh[np.ix_(subcort_indices, subcort_indices)]
            
            # 1. Overall subcortical metrics
            subcort_efficiency, subcort_path = compute_efficiency_pathlength(
                subcort_matrix,
                use_bct=use_bct
            )
            
            metrics = {
                'subject': subj,
                'group': group_name,
                'subcort_global_efficiency': subcort_efficiency,
                'subcort_char_path_length': subcort_path
            }
            
            # 2. Category-specific metrics
            if parcel_to_category is not None:
                categories = set(parcel_to_category.values())
                for category in categories:
                    # Extract submatrix for this category
                    cat_indices = [i for i in range(360, 424) 
                                 if parcel_to_category.get(i+1) == category]
                    if len(cat_indices) < 2:
                        continue
                    
                    # Convert to 0-indexed for subcortical matrix
                    cat_indices_subcort = [i - 360 for i in cat_indices]
                    cat_submatrix = subcort_matrix[np.ix_(cat_indices_subcort, cat_indices_subcort)]
                    
                    cat_efficiency, cat_path = compute_efficiency_pathlength(
                        cat_submatrix,
                        use_bct=use_bct
                    )
                    
                    cat_key = category.replace(" ", "_").lower()
                    metrics[f'category_{cat_key}_efficiency'] = cat_efficiency
                    metrics[f'category_{cat_key}_pathlength'] = cat_path
            
            efficiency_metrics.append(metrics)
    
    return pd.DataFrame(efficiency_metrics)


def run_efficiency_statistical_tests(
    metrics_df: pd.DataFrame,
    metric_col: str,
    groups: Dict[str, List[str]],
    title: str,
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive statistical tests for efficiency/path length metrics.
    Similar to run_connectivity_statistical_tests but adapted for efficiency metrics.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with efficiency/path length metrics
    metric_col : str
        Column name of the metric to test
    groups : dict
        Dictionary mapping group names to subject lists
    title : str
        Title for the statistical test section
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    results : dict
        Dictionary with statistical test results
    """
    # Use base function with efficiency-specific parameters
    results = _run_statistical_tests_base(
        metrics_df=metrics_df,
        metric_col=metric_col,
        groups=groups,
        title=title,
        verbose=verbose,
        value_format='.6f'  # Fixed decimal format for efficiency/path length values
    )
    
    # Add backward-compatible aliases for post-hoc results (if needed by existing code)
    if 'anova_posthoc' in results and results['anova_posthoc']:
        # Create simplified posthoc_anova structure for backward compatibility
        results['posthoc_anova'] = {
            'comparisons': [(item['group1'], item['group2']) for item in results['anova_posthoc']],
            'pvalues': [item['pvalue_uncorrected'] for item in results['anova_posthoc']],
            'pvalues_corrected': [item['pvalue_corrected'] for item in results['anova_posthoc']]
        }
    
    if 'kruskal_wallis_posthoc' in results and results['kruskal_wallis_posthoc']:
        # Create simplified posthoc_kw structure for backward compatibility
        results['posthoc_kw'] = {
            'comparisons': [(item['group1'], item['group2']) for item in results['kruskal_wallis_posthoc']],
            'pvalues': [item['pvalue_uncorrected'] for item in results['kruskal_wallis_posthoc']],
            'pvalues_corrected': [item['pvalue_corrected'] for item in results['kruskal_wallis_posthoc']],
            'u_statistics': [item['u_statistic'] for item in results['kruskal_wallis_posthoc']]
        }
    
    return results


def plot_efficiency_pathlength_results(
    metrics_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    connectome_type: str,
    region_type: str = 'whole_brain',
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create visualization plots for efficiency/path length results.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with metrics from compute_efficiency_pathlength_metrics()
    groups : dict
        Dictionary of groups
    connectome_type : str
        Connectome type (e.g., 'SC_sift2_sizecorr', 'MEAN_FA', 'MEAN_MD')
    region_type : str
        'whole_brain' or 'subcortical'
    save_path : str or Path, optional
        Path to save figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Define group order and colors
    group_order = ['Healthy Controls', 'ICU', 'ICU Delirium']
    group_order = [g for g in group_order if g in groups or g in metrics_df['group'].unique()]
    colors = get_group_colors(group_order)
    
    if region_type == 'whole_brain':
        # Whole-brain visualization: 4 plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Efficiency/Path Length Analysis: {connectome_type} (Whole-Brain)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Global efficiency by group
        ax1 = axes[0, 0]
        data_to_plot = [metrics_df[metrics_df['group'] == g]['global_efficiency'].dropna().values 
                        for g in group_order]
        bp1 = ax1.boxplot(data_to_plot, labels=group_order, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Global Efficiency', fontsize=12)
        ax1.set_title('Global Efficiency by Group', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Characteristic path length by group
        ax2 = axes[0, 1]
        data_to_plot = [metrics_df[metrics_df['group'] == g]['char_path_length'].dropna().values 
                        for g in group_order]
        bp2 = ax2.boxplot(data_to_plot, labels=group_order, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Characteristic Path Length', fontsize=12)
        ax2.set_title('Characteristic Path Length by Group', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Efficiency vs Path Length scatter
        ax3 = axes[1, 0]
        for i, group_name in enumerate(group_order):
            group_data = metrics_df[metrics_df['group'] == group_name]
            valid = group_data[['global_efficiency', 'char_path_length']].dropna()
            if len(valid) > 0:
                ax3.scatter(valid['char_path_length'], valid['global_efficiency'],
                           label=group_name, color=colors[i], alpha=0.6, s=60)
        ax3.set_xlabel('Characteristic Path Length', fontsize=11)
        ax3.set_ylabel('Global Efficiency', fontsize=11)
        ax3.set_title('Efficiency vs Path Length (Inverse Relationship)', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Network-specific efficiency
        ax4 = axes[1, 1]
        yeo7_efficiency_cols = [col for col in metrics_df.columns 
                                if col.endswith('_efficiency') and 'yeo7' in col]
        yeo7_net_ids = sorted({int(col.split('_')[2]) for col in yeo7_efficiency_cols})
        
        if len(yeo7_net_ids) > 0:
            yeo7_names = {
                0: 'Subcortical/Unassigned',
                1: 'Visual',
                2: 'Somatomotor',
                3: 'Dorsal Attention',
                4: 'Ventral Attention',
                5: 'Limbic',
                6: 'Frontoparietal',
                7: 'Default Mode'
            }
            
            net_means = {net_id: [] for net_id in yeo7_net_ids}
            for net_id in yeo7_net_ids:
                col_name = f'yeo7_net_{net_id}_efficiency'
                if col_name in metrics_df.columns:
                    for group_name in group_order:
                        group_vals = metrics_df[metrics_df['group'] == group_name][col_name].dropna()
                        if len(group_vals) > 0:
                            net_means[net_id].append(group_vals.mean())
                        else:
                            net_means[net_id].append(np.nan)
            
            x = np.arange(len(yeo7_net_ids))
            width = 0.25
            for i, group_name in enumerate(group_order):
                means = [net_means[net_id][i] if i < len(net_means[net_id]) else np.nan 
                        for net_id in yeo7_net_ids]
                ax4.bar(x + i*width, means, width, label=group_name, color=colors[i], alpha=0.7)
            
            ax4.set_xlabel('Yeo7 Network', fontsize=11)
            ax4.set_ylabel('Mean Network Efficiency', fontsize=11)
            ax4.set_title('Network-Specific Efficiency', fontsize=13, fontweight='bold')
            ax4.set_xticks(x + width)
            ax4.set_xticklabels([yeo7_names.get(nid, f'Network {nid}') for nid in yeo7_net_ids], 
                               rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
    else:  # subcortical
        # Subcortical visualization: 2 plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Efficiency/Path Length Analysis: {connectome_type} (Subcortical)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Subcortical global efficiency by group
        ax1 = axes[0]
        data_to_plot = [metrics_df[metrics_df['group'] == g]['subcort_global_efficiency'].dropna().values 
                        for g in group_order]
        bp1 = ax1.boxplot(data_to_plot, labels=group_order, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Subcortical Global Efficiency', fontsize=12)
        ax1.set_title('Subcortical Global Efficiency by Group', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Subcortical characteristic path length by group
        ax2 = axes[1]
        data_to_plot = [metrics_df[metrics_df['group'] == g]['subcort_char_path_length'].dropna().values 
                        for g in group_order]
        bp2 = ax2.boxplot(data_to_plot, labels=group_order, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Subcortical Characteristic Path Length', fontsize=12)
        ax2.set_title('Subcortical Characteristic Path Length by Group', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def run_full_efficiency_analysis(
    preproc_dir: Union[str, Path],
    connectome_type: str,
    groups: Dict[str, List[str]],
    region_type: str = 'whole_brain',
    save_dir: Optional[Union[str, Path]] = None,
    target_density: float = 0.15,
    use_bct: bool = False,
    verbose: bool = True,
    parcel_to_yeo7: Optional[Dict[int, int]] = None,
    parcel_to_category: Optional[Dict[int, str]] = None
) -> Dict:
    """
    High-level function that runs the complete efficiency/path length analysis pipeline.
    
    Parameters:
    -----------
    preproc_dir : str or Path
        Preprocessing directory path
    connectome_type : str
        Connectome type to analyze (e.g., 'SC_sift2_sizecorr', 'MEAN_FA', 'MEAN_MD')
    groups : dict
        Dictionary of groups mapping group names to subject lists
    region_type : str
        'whole_brain' or 'subcortical'
    save_dir : str or Path, optional
        Directory to save results. If None, saves to preproc_dir/results/
    target_density : float
        Target density for thresholding (e.g., 0.15 for top 15% of edges)
    use_bct : bool
        If True, use bctpy; otherwise use NetworkX
    verbose : bool
        Whether to print progress
    parcel_to_yeo7 : dict, optional
        Mapping from parcel_id to Yeo7 network ID (required for whole_brain)
    parcel_to_category : dict, optional
        Mapping from parcel_id to subcortical category (required for subcortical)
    
    Returns:
    --------
    results : dict
        Dictionary with keys:
        - 'metrics_df': DataFrame with computed metrics
        - 'statistical_results': Dictionary of statistical test results
        - 'figure': matplotlib Figure object
        - 'save_paths': Dictionary of saved file paths
    """
    preproc_dir = Path(preproc_dir)
    if save_dir is None:
        save_dir = preproc_dir / 'results'
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*80)
        print(f"EFFICIENCY/PATH LENGTH ANALYSIS: {connectome_type} ({region_type})")
        print("="*80)
    
    # Load connectomes
    if verbose:
        print(f"\nLoading connectomes of type: {connectome_type}")
    connectomes_dict = load_all_connectomes(preproc_dir, connectome_type)
    
    if len(connectomes_dict) == 0:
        raise ValueError(f"No connectomes found for type: {connectome_type}")
    
    if verbose:
        print(f"Loaded {len(connectomes_dict)} connectomes")
    
    # Compute metrics
    if verbose:
        print(f"\nComputing efficiency/path length metrics ({region_type})...")
        print(f"  Global normalization: YES (normalize all subjects by global mean weight before thresholding)")
        print(f"  Density thresholding: {target_density} (keep top {target_density*100}% of edges)")
        print(f"  Note: This ensures fair topology comparison by removing weight scale differences between groups")
    
    metrics_df = compute_efficiency_pathlength_metrics(
        connectomes_dict=connectomes_dict,
        groups=groups,
        region_type=region_type,
        target_density=target_density,
        use_bct=use_bct,
        parcel_to_yeo7=parcel_to_yeo7,
        parcel_to_category=parcel_to_category
    )
    
    if verbose:
        print(f"Computed metrics for {len(metrics_df)} subjects")
        print(f"Groups: {metrics_df['group'].value_counts().to_dict()}")
    
    # Run statistical tests
    if verbose:
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
    
    statistical_results = {}
    
    if region_type == 'whole_brain':
        # Test global_efficiency, char_path_length, and each network
        test_metrics = ['global_efficiency', 'char_path_length']
        yeo7_efficiency_cols = [col for col in metrics_df.columns 
                                if col.endswith('_efficiency') and 'yeo7' in col]
        test_metrics.extend(yeo7_efficiency_cols)
    else:  # subcortical
        # Test subcort_global_efficiency, subcort_char_path_length, and each category
        test_metrics = ['subcort_global_efficiency', 'subcort_char_path_length']
        category_efficiency_cols = [col for col in metrics_df.columns 
                                    if col.endswith('_efficiency') and 'category' in col]
        test_metrics.extend(category_efficiency_cols)
    
    for metric_name in test_metrics:
        if metric_name in metrics_df.columns:
            if verbose:
                print(f"\n{'='*80}")
            results = run_efficiency_statistical_tests(
                metrics_df=metrics_df,
                metric_col=metric_name,
                groups=groups,
                title=metric_name.upper().replace('_', ' '),
                verbose=verbose
            )
            statistical_results[metric_name] = results
    
    # Create visualization
    if verbose:
        print("\n" + "="*80)
        print("VISUALIZATION")
        print("="*80)
    
    fig_save_path = save_dir / f'efficiency_pathlength_{region_type}_{connectome_type}.png'
    fig = plot_efficiency_pathlength_results(
        metrics_df=metrics_df,
        groups=groups,
        connectome_type=connectome_type,
        region_type=region_type,
        save_path=fig_save_path
    )
    
    # Save metrics DataFrame
    metrics_save_path = save_dir / f'efficiency_pathlength_{region_type}_{connectome_type}.csv'
    metrics_df.to_csv(metrics_save_path, index=False)
    if verbose:
        print(f"Metrics saved to {metrics_save_path}")
    
    return {
        'metrics_df': metrics_df,
        'statistical_results': statistical_results,
        'figure': fig,
        'save_paths': {
            'figure': fig_save_path,
            'metrics': metrics_save_path
        }
    }


# ============================================================================
# BOOTSTRAP ANALYSIS FUNCTIONS
# ============================================================================

def bootstrap_global_network_metrics(
    connectomes_dict: Dict[str, np.ndarray],
    groups: Dict[str, List[str]],
    metric_type: str = 'connectivity_strength',
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
    region_type: str = 'whole_brain',
    target_density: Optional[float] = None,
    use_bct: bool = False,
    parcel_to_yeo7: Optional[Dict[int, int]] = None,
    subcort_df: Optional[pd.DataFrame] = None,
    parcel_to_category: Optional[Dict[int, str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Perform stratified bootstrap resampling to compute confidence intervals for global network metrics.
    
    This function resamples subjects within each group (stratified bootstrap) and recomputes
    metrics B times to estimate uncertainty. Particularly useful for small sample sizes (e.g., n=5).
    
    Parameters:
    -----------
    connectomes_dict : dict
        Dictionary of connectomes (from load_all_connectomes()), keys are subject IDs with 'sub-' prefix
    groups : dict
        Dictionary mapping group names to subject lists (without 'sub-' prefix)
    metric_type : str
        Type of metrics to compute: 'connectivity_strength' or 'efficiency_pathlength'
    n_bootstrap : int
        Number of bootstrap iterations (default: 5000)
    confidence_level : float
        Confidence level for intervals (default: 0.95 for 95% CI)
    random_seed : int, optional
        Random seed for reproducibility
    region_type : str
        'whole_brain' or 'subcortical'
    target_density : float, optional
        Target density for thresholding (for efficiency_pathlength only)
    use_bct : bool
        If True, use bctpy; otherwise use NetworkX (for efficiency_pathlength only)
    parcel_to_yeo7 : dict, optional
        Mapping dictionary from parcel_id (1-430) to Yeo7 network ID (for whole-brain only)
    subcort_df : pd.DataFrame, optional
        DataFrame with subcortical parcel info (for subcortical only)
    parcel_to_category : dict, optional
        Mapping from parcel_id to subcortical category (for subcortical only)
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    bootstrap_results : dict
        Dictionary containing:
        - 'bootstrap_distributions': dict of metric_name -> list of bootstrap values
        - 'group_means': dict of metric_name -> dict of group_name -> mean value
        - 'group_cis': dict of metric_name -> dict of group_name -> (lower, upper) CI
        - 'group_differences': dict of comparison -> dict of metric_name -> (mean_diff, ci_lower, ci_upper, p_value)
        - 'sign_flips': dict of comparison -> dict of metric_name -> proportion of sign flips
        - 'n_bootstrap': number of bootstrap iterations
        - 'confidence_level': confidence level used
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Prepare groups: filter subjects that exist in connectomes_dict
    valid_groups = {}
    for group_name, group_subjects in groups.items():
        valid_subjects = []
        for subj in group_subjects:
            subj_key = f'sub-{subj}' if not subj.startswith('sub-') else subj
            if subj_key in connectomes_dict:
                valid_subjects.append(subj)
        if len(valid_subjects) > 0:
            valid_groups[group_name] = valid_subjects
    
    if len(valid_groups) < 2:
        raise ValueError("Need at least 2 groups with valid subjects for bootstrap analysis")
    
    if verbose:
        print("="*80)
        print(f"BOOTSTRAP ANALYSIS: {metric_type.upper()} ({region_type})")
        print("="*80)
        print(f"Groups: {list(valid_groups.keys())}")
        for group_name, subjects in valid_groups.items():
            print(f"  {group_name}: {len(subjects)} subjects")
        print(f"Bootstrap iterations: {n_bootstrap}")
        print(f"Confidence level: {confidence_level*100}%")
        print()
    
    # Initialize storage for bootstrap distributions
    bootstrap_distributions = {}
    all_metric_names = None
    
    # Run bootstrap iterations
    if verbose:
        print("Running bootstrap iterations...")
        print_progress_every = max(1, n_bootstrap // 20)  # Print every 5%
    
    for b in range(n_bootstrap):
        # Resample subjects within each group (stratified bootstrap)
        resampled_groups = {}
        for group_name, subjects in valid_groups.items():
            # Sample with replacement, maintaining group size
            resampled_subjects = random.choices(subjects, k=len(subjects))
            resampled_groups[group_name] = resampled_subjects
        
        # Create resampled connectomes_dict
        resampled_connectomes = {}
        for group_name, resampled_subjects in resampled_groups.items():
            for subj in resampled_subjects:
                subj_key = f'sub-{subj}' if not subj.startswith('sub-') else subj
                if subj_key in connectomes_dict:
                    # Use original connectome (resampling is at subject level, not connectome level)
                    resampled_connectomes[subj_key] = connectomes_dict[subj_key]
        
        # Compute metrics for this bootstrap sample
        if metric_type == 'connectivity_strength':
            metrics_df = compute_connectivity_strength_metrics(
                connectomes_dict=resampled_connectomes,
                groups=resampled_groups,
                region_type=region_type,
                parcel_to_yeo7=parcel_to_yeo7,
                subcort_df=subcort_df,
                parcel_to_category=parcel_to_category
            )
        elif metric_type == 'efficiency_pathlength':
            metrics_df = compute_efficiency_pathlength_metrics(
                connectomes_dict=resampled_connectomes,
                groups=resampled_groups,
                region_type=region_type,
                target_density=target_density,
                use_bct=use_bct,
                parcel_to_yeo7=parcel_to_yeo7,
                parcel_to_category=parcel_to_category
            )
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}. Must be 'connectivity_strength' or 'efficiency_pathlength'")
        
        if len(metrics_df) == 0:
            continue
        
        # Store metric names from first iteration
        if all_metric_names is None:
            # Exclude subject and group columns
            all_metric_names = [col for col in metrics_df.columns 
                              if col not in ['subject', 'group']]
        
        # Compute group means for this bootstrap iteration
        for metric_name in all_metric_names:
            if metric_name not in bootstrap_distributions:
                bootstrap_distributions[metric_name] = {}
            
            for group_name in valid_groups.keys():
                group_data = metrics_df[metrics_df['group'] == group_name][metric_name].dropna()
                if len(group_data) > 0:
                    group_mean = group_data.mean()
                    if group_name not in bootstrap_distributions[metric_name]:
                        bootstrap_distributions[metric_name][group_name] = []
                    bootstrap_distributions[metric_name][group_name].append(group_mean)
        
        if verbose and (b + 1) % print_progress_every == 0:
            print(f"  Completed {b + 1}/{n_bootstrap} iterations ({(b+1)/n_bootstrap*100:.1f}%)")
    
    if verbose:
        print(f"✓ Completed all {n_bootstrap} bootstrap iterations\n")
    
    # Compute confidence intervals and summary statistics
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    group_means = {}
    group_cis = {}
    
    for metric_name in all_metric_names:
        group_means[metric_name] = {}
        group_cis[metric_name] = {}
        
        for group_name in valid_groups.keys():
            if group_name in bootstrap_distributions[metric_name]:
                values = np.array(bootstrap_distributions[metric_name][group_name])
                group_means[metric_name][group_name] = np.mean(values)
                ci_lower = np.percentile(values, lower_percentile)
                ci_upper = np.percentile(values, upper_percentile)
                group_cis[metric_name][group_name] = (ci_lower, ci_upper)
    
    # Compute group differences and empirical p-values
    group_names_list = list(valid_groups.keys())
    group_differences = {}
    sign_flips = {}
    
    for i, group1 in enumerate(group_names_list):
        for group2 in group_names_list[i+1:]:
            comparison = f"{group1}_vs_{group2}"
            group_differences[comparison] = {}
            sign_flips[comparison] = {}
            
            for metric_name in all_metric_names:
                if (group1 in bootstrap_distributions[metric_name] and 
                    group2 in bootstrap_distributions[metric_name]):
                    
                    values1 = np.array(bootstrap_distributions[metric_name][group1])
                    values2 = np.array(bootstrap_distributions[metric_name][group2])
                    
                    # Compute differences
                    differences = values1 - values2
                    mean_diff = np.mean(differences)
                    ci_lower = np.percentile(differences, lower_percentile)
                    ci_upper = np.percentile(differences, upper_percentile)
                    
                    # Empirical p-value: proportion of differences that cross zero
                    # Two-tailed: proportion where |difference| >= |observed_mean_diff|
                    observed_abs_diff = abs(mean_diff)
                    p_value = np.mean(np.abs(differences) >= observed_abs_diff)
                    
                    # Sign flips: proportion where difference has opposite sign from mean
                    mean_sign = np.sign(mean_diff)
                    if mean_sign != 0:
                        flip_proportion = np.mean(np.sign(differences) != mean_sign)
                    else:
                        flip_proportion = 0.5  # If mean is exactly zero, half will be opposite
                    
                    group_differences[comparison][metric_name] = {
                        'mean_diff': mean_diff,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_value': p_value
                    }
                    sign_flips[comparison][metric_name] = flip_proportion
    
    return {
        'bootstrap_distributions': bootstrap_distributions,
        'group_means': group_means,
        'group_cis': group_cis,
        'group_differences': group_differences,
        'sign_flips': sign_flips,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'metric_type': metric_type,
        'region_type': region_type
    }


def plot_bootstrap_results(
    bootstrap_results: Dict,
    groups: Dict[str, List[str]],
    connectome_type: str,
    metric_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create visualization of bootstrap results with confidence intervals.
    
    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap_global_network_metrics()
    groups : dict
        Dictionary mapping group names to subject lists
    connectome_type : str
        Type of connectome (for title)
    metric_names : list, optional
        List of metric names to plot. If None, plots all metrics.
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    group_means = bootstrap_results['group_means']
    group_cis = bootstrap_results['group_cis']
    metric_type = bootstrap_results.get('metric_type', 'metrics')
    region_type = bootstrap_results.get('region_type', 'whole_brain')
    
    if metric_names is None:
        metric_names = list(group_means.keys())
    
    # Filter to available metrics
    metric_names = [m for m in metric_names if m in group_means]
    
    if len(metric_names) == 0:
        raise ValueError("No valid metrics to plot")
    
    # Determine layout
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    group_names = list(groups.keys())
    color_list = get_group_colors(group_names)
    colors = dict(zip(group_names, color_list))
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        
        # Prepare data for plotting
        x_pos = np.arange(len(group_names))
        means = [group_means[metric_name].get(g, np.nan) for g in group_names]
        ci_lowers = [group_cis[metric_name].get(g, (np.nan, np.nan))[0] for g in group_names]
        ci_uppers = [group_cis[metric_name].get(g, (np.nan, np.nan))[1] for g in group_names]
        errors_lower = [m - cl for m, cl in zip(means, ci_lowers)]
        errors_upper = [cu - m for m, cu in zip(means, ci_uppers)]
        
        # Plot bars with error bars
        bars = ax.bar(x_pos, means, yerr=[errors_lower, errors_upper], 
                     capsize=5, color=[colors.get(g, 'gray') for g in group_names],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            if not np.isnan(mean_val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean_val:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'{metric_name.replace("_", " ").title()}\n(95% CI)', 
                    fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    # Main title
    title = f'Bootstrap Results: {metric_type.replace("_", " ").title()} ({region_type.replace("_", " ").title()})\n{connectome_type}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved bootstrap figure to {save_path}")
    
    return fig


def save_bootstrap_summary(
    bootstrap_results: Dict,
    save_path: Union[str, Path],
    verbose: bool = True
):
    """
    Save bootstrap results summary to CSV files.
    
    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap_global_network_metrics()
    save_path : str or Path
        Base path for saving (will create multiple files)
    verbose : bool
        Whether to print save messages
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    group_means = bootstrap_results['group_means']
    group_cis = bootstrap_results['group_cis']
    group_differences = bootstrap_results['group_differences']
    sign_flips = bootstrap_results['sign_flips']
    
    # 1. Group means and CIs
    means_data = []
    for metric_name, group_data in group_means.items():
        for group_name, mean_val in group_data.items():
            ci_lower, ci_upper = group_cis[metric_name][group_name]
            means_data.append({
                'metric': metric_name,
                'group': group_name,
                'mean': mean_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            })
    
    means_df = pd.DataFrame(means_data)
    means_path = save_path.parent / f"{save_path.stem}_group_means.csv"
    means_df.to_csv(means_path, index=False)
    if verbose:
        print(f"✓ Saved group means to {means_path}")
    
    # 2. Group differences
    diff_data = []
    for comparison, metric_data in group_differences.items():
        for metric_name, diff_info in metric_data.items():
            diff_data.append({
                'comparison': comparison,
                'metric': metric_name,
                'mean_difference': diff_info['mean_diff'],
                'ci_lower': diff_info['ci_lower'],
                'ci_upper': diff_info['ci_upper'],
                'p_value': diff_info['p_value']
            })
    
    if diff_data:
        diff_df = pd.DataFrame(diff_data)
        diff_path = save_path.parent / f"{save_path.stem}_group_differences.csv"
        diff_df.to_csv(diff_path, index=False)
        if verbose:
            print(f"✓ Saved group differences to {diff_path}")
    
    # 3. Sign flips
    flip_data = []
    for comparison, metric_data in sign_flips.items():
        for metric_name, flip_prop in metric_data.items():
            flip_data.append({
                'comparison': comparison,
                'metric': metric_name,
                'sign_flip_proportion': flip_prop
            })
    
    if flip_data:
        flip_df = pd.DataFrame(flip_data)
        flip_path = save_path.parent / f"{save_path.stem}_sign_flips.csv"
        flip_df.to_csv(flip_path, index=False)
        if verbose:
            print(f"✓ Saved sign flips to {flip_path}")


# ============================================================================
# CONNECTOME ORDERING AND VISUALIZATION FUNCTIONS
# ============================================================================

def _safe_fiedler_order(Wb: np.ndarray) -> np.ndarray:
    """
    Compute Fiedler ordering (spectral ordering) for a block matrix.
    
    Parameters:
    -----------
    Wb : np.ndarray
        Block connectivity matrix
        
    Returns:
    --------
    order : np.ndarray
        Indices that order the nodes by Fiedler vector
    """
    Wb = np.array(Wb, dtype=float)
    Wb = 0.5 * (Wb + Wb.T)  # Ensure symmetric
    np.fill_diagonal(Wb, 0.0)
    
    d = Wb.sum(1)
    if np.allclose(d, 0):
        return np.arange(Wb.shape[0])
    
    Dm12 = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(Wb.shape[0]) - (Dm12[:, None] * Wb * Dm12[None, :])
    
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        _, vecs = spla.eigsh(sp.csr_matrix(L), k=2, which='SM')
        return np.argsort(vecs[:, 1])
    except Exception:
        # Fallback: order by degree
        return np.argsort(-d)


def build_cortical_order_LH_RH(
    M: np.ndarray,
    A7: Union[np.ndarray, Dict[int, int], pd.DataFrame],
    yeo7_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build ordering for cortical parcels: LH first, then RH, ordered by Yeo7 networks.
    Uses spectral/Fiedler ordering within each hemisphere-network block.
    
    Parameters:
    -----------
    M : np.ndarray
        Connectome matrix (360×360 cortical or 430×430 full)
    A7 : np.ndarray, dict, or pd.DataFrame
        Yeo7 assignment. Can be:
        - 1D array (360,) with values 1-7
        - Dict mapping parcel_id (1-360) to network (1-7)
        - DataFrame with 'parcel_id' and 'yeo7_network' columns
    yeo7_names : list
        List of 7 Yeo7 network names in order [Network1, Network2, ..., Network7]
        e.g., ['Visual', 'Somatomotor', 'DorsalAttn', 'VentralAttn', 'Limbic', 'FrontoParietal', 'Default']
        
    Returns:
    --------
    order : np.ndarray
        0-indexed parcel indices (length 360)
    breaks : np.ndarray
        Block boundaries (cumulative sizes)
    block_labels : list
        Block labels like ['LH Visual', 'LH Somatomotor', ..., 'RH Default']
    """
    # Extract cortical submatrix if needed
    if M.shape[0] == 430:
        M_cortical = M[:360, :360]
    elif M.shape[0] == 360:
        M_cortical = M.copy()
    else:
        raise ValueError(f"Expected matrix shape (360, 360) or (430, 430), got {M.shape}")
    
    M_cortical = prepare_matrix(M_cortical)
    
    # Convert A7 to dict if needed
    if isinstance(A7, pd.DataFrame):
        parcel_to_yeo7 = dict(zip(A7['parcel_id'], A7['yeo7_network']))
    elif isinstance(A7, np.ndarray):
        parcel_to_yeo7 = {i+1: int(A7[i]) for i in range(len(A7)) if A7[i] > 0}
    elif isinstance(A7, dict):
        parcel_to_yeo7 = A7
    else:
        raise ValueError(f"Unsupported A7 type: {type(A7)}")
    
    # Determine hemisphere: parcels 1-180 are LH, 181-360 are RH
    hemi = np.zeros(360, dtype=int)
    hemi[180:] = 1  # 0=LH, 1=RH
    
    # Get network labels for each parcel (1-indexed parcel IDs)
    net_labels = np.zeros(360, dtype=int)
    for i in range(360):
        parcel_id = i + 1  # Convert to 1-indexed
        if parcel_id in parcel_to_yeo7:
            net_labels[i] = int(parcel_to_yeo7[parcel_id])
    
    # Build blocks: hemisphere-major, then network
    blocks = []
    block_labels = []
    
    for h in (0, 1):  # LH first, then RH
        for k in range(1, 8):  # Networks 1-7
            idx = np.where((hemi == h) & (net_labels == k))[0]
            if idx.size == 0:
                continue
            
            # Get submatrix for this block
            Wb = M_cortical[np.ix_(idx, idx)]
            
            # Spectral ordering within block
            o = _safe_fiedler_order(Wb)
            blocks.append(idx[o])
            block_labels.append(('LH' if h == 0 else 'RH') + ' ' + yeo7_names[k - 1])
    
    # Concatenate all blocks
    if blocks:
        order = np.concatenate(blocks)
    else:
        order = np.arange(360)
    
    # Validate order
    assert len(order) == 360 and len(set(order)) == 360
    assert order.min() == 0 and order.max() == 359
    
    # Compute breaks
    breaks = np.cumsum([len(b) for b in blocks])
    
    return order, breaks, block_labels


def build_subcortical_order_by_category(
    M: np.ndarray,
    parcel_to_category: Dict[int, str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build ordering for subcortical parcels by category with LH/RH separation.
    Orders by: hemisphere first (LH, then RH), then category within each hemisphere.
    Uses spectral/Fiedler ordering within each hemisphere-category block.
    
    Hemisphere organization:
    - LH: parcels 361-392 (subcortical) + 430 (cerebellum)
    - RH: parcels 393-424 (subcortical) + 429 (cerebellum)
    - No hemisphere: parcels 425-428 (brainstem)
    
    Parameters:
    -----------
    M : np.ndarray
        Connectome matrix (64×64 subcortical or 430×430 full)
    parcel_to_category : dict
        Maps parcel_id (361-430) to category name
        Categories: 'Thalamus', 'Basal Ganglia', 'Brainstem', 'Amygdala', 'Cerebellum'
        
    Returns:
    --------
    order : np.ndarray
        0-indexed parcel indices within subcortical (length 64, values 0-63)
    breaks : np.ndarray
        Block boundaries (cumulative sizes)
    block_labels : list
        Block labels like ['LH Thalamus', 'LH Basal Ganglia', ..., 'RH Cerebellum', 'Brainstem']
    """
    # Extract subcortical submatrix if needed
    # In a 430×430 matrix, subcortical parcels 361-424 need to be extracted
    # accounting for removed parcels 390 and 423
    # The result is a 62×62 matrix (64 parcels - 2 excluded = 62)
    if M.shape[0] == 430:
        # Build list of matrix indices for subcortical parcels 361-424
        # Parcels 361-389: positions 360-388 (29 parcels)
        # Parcel 390: REMOVED (was position 389)
        # Parcels 391-422: positions 389-420 (32 parcels, shifted down by 1)
        # Parcel 423: REMOVED (was position 422)
        # Parcel 424: position 421 (1 parcel, shifted down by 2)
        subcort_indices = []
        for parcel_id in range(361, 425):  # 361-424
            if parcel_id == 390:
                continue  # Skip removed parcel
            elif parcel_id == 423:
                continue  # Skip removed parcel
            elif parcel_id < 390:
                # Parcels 361-389: direct mapping
                subcort_indices.append(parcel_id - 1)  # 0-indexed: 360-388
            elif parcel_id < 423:
                # Parcels 391-422: shifted down by 1 (390 was removed)
                subcort_indices.append(parcel_id - 2)  # 0-indexed: 389-420
            else:  # parcel_id == 424
                # Parcel 424: shifted down by 2 (390 and 423 were removed)
                subcort_indices.append(parcel_id - 3)  # 0-indexed: 421
        
        # Extract submatrix using these indices
        subcort_indices = np.array(subcort_indices)
        M_subcort = M[np.ix_(subcort_indices, subcort_indices)]
        
        # Create mapping from parcel_id to position in M_subcort (0-61)
        # Position corresponds to order in the extracted matrix
        parcel_to_subcort_pos = {}
        pos = 0
        for parcel_id in range(361, 425):
            if parcel_id not in [390, 423]:
                parcel_to_subcort_pos[parcel_id] = pos
                pos += 1
        
    elif M.shape[0] == 62:
        # Already a correctly extracted 62×62 subcortical matrix
        M_subcort = M.copy()
        # Create mapping: parcel 361 -> 0, 362 -> 1, etc. (skipping 390 and 423)
        # This should map to positions 0-61 (62 parcels total)
        parcel_to_subcort_pos = {}
        pos = 0
        for parcel_id in range(361, 425):
            if parcel_id not in [390, 423]:
                parcel_to_subcort_pos[parcel_id] = pos
                pos += 1
        
        # Validate that the mapping size matches the matrix size
        if len(parcel_to_subcort_pos) != M_subcort.shape[0]:
            raise ValueError(
                f"Subcortical matrix size mismatch: matrix is {M_subcort.shape[0]}×{M_subcort.shape[1]}, "
                f"but expected {len(parcel_to_subcort_pos)}×{len(parcel_to_subcort_pos)} "
                f"(62×62 after removing parcels 390 and 423)."
            )
    elif M.shape[0] == 64:
        # Already a subcortical matrix - but it might be 64×64 from old code
        # or correctly extracted as 62×62. Check actual size.
        if M.shape[0] == 64 and M.shape[1] == 64:
            # Old 64×64 format - assume it has all 64 positions
            # But we only have 62 valid parcels, so we need to map carefully
            # For now, treat as if it's correctly extracted (62 parcels)
            # If it's actually 64×64 with 2 invalid positions, we'll handle that
            M_subcort = M.copy()
        else:
            M_subcort = M.copy()
        
        # Create mapping: parcel 361 -> 0, 362 -> 1, etc. (skipping 390 and 423)
        # This should map to positions 0-61 (62 parcels total)
        parcel_to_subcort_pos = {}
        pos = 0
        for parcel_id in range(361, 425):
            if parcel_id not in [390, 423]:
                parcel_to_subcort_pos[parcel_id] = pos
                pos += 1
        
        # Validate that the mapping size matches the matrix size
        expected_size = len(parcel_to_subcort_pos)  # Should be 62
        actual_size = M_subcort.shape[0]
        if actual_size != expected_size:
            if actual_size == 64:
                # This is an old 64×64 matrix - we need to extract only the valid 62×62 submatrix
                # The invalid positions are at indices 29 and 62 (corresponding to parcels 390 and 423)
                # But we don't know the exact positions without more context
                # For now, raise a clear error
                raise ValueError(
                    f"Subcortical matrix is 64×64 (old format) but expected 62×62 "
                    f"(after removing parcels 390 and 423). "
                    f"Please use a 430×430 full matrix instead, or regenerate connectomes to 430×430 format."
                )
            else:
                raise ValueError(
                    f"Subcortical matrix size mismatch: matrix is {actual_size}×{actual_size}, "
                    f"but expected {expected_size}×{expected_size} "
                    f"(62×62 after removing parcels 390 and 423)."
                )
    else:
        raise ValueError(
            f"Expected matrix shape (62, 62), (64, 64), or (430, 430), got {M.shape}. "
            f"Note: 62×62 is the correct size after removing parcels 390 and 423."
        )
    
    M_subcort = prepare_matrix(M_subcort)
    
    # Category order (excluding Brainstem and Cerebellum which are handled separately)
    category_order = ['Thalamus', 'Basal Ganglia', 'Amygdala']
    
    # Track which parcels have been assigned
    assigned_parcels = set()
    
    # Build blocks: hemisphere-major, then category
    blocks = []
    block_labels = []
    
    # Helper function to determine hemisphere from parcel_id
    # Note: Subcortical region is 361-424 (64 parcels) after removing parcels 390 and 423
    def get_hemisphere(parcel_id):
        """Return 'LH', 'RH', or None for no hemisphere (brainstem)"""
        # Subcortical parcels 361-424 (after removing 390 and 423)
        # LH: 361-389 (excluding 390)
        # RH: 391-422 (excluding 423), 424
        if 361 <= parcel_id <= 389:
            return 'LH'
        elif 391 <= parcel_id <= 424:
            return 'RH'
        elif 425 <= parcel_id <= 428:
            return None  # Brainstem has no hemisphere (not in subcortical matrix)
        elif parcel_id == 429:
            return 'RH'  # R-cerebellum (not in subcortical matrix)
        elif parcel_id == 430:
            return 'LH'  # L-cerebellum (not in subcortical matrix)
        else:
            return None
    
    # Use the mapping created during matrix extraction
    def parcel_id_to_subcort_idx(parcel_id):
        """Convert parcel ID to position in M_subcort matrix."""
        if parcel_id in [390, 423]:
            raise ValueError(f"Parcel {parcel_id} is excluded")
        if parcel_id < 361 or parcel_id > 424:
            raise ValueError(f"Parcel {parcel_id} is not in subcortical range (361-424)")
        if parcel_id not in parcel_to_subcort_pos:
            raise ValueError(
                f"Parcel {parcel_id} is in range but not in mapping. "
                f"Mapping has {len(parcel_to_subcort_pos)} parcels. "
                f"Matrix size: {M_subcort.shape[0]}×{M_subcort.shape[1]}"
            )
        idx = parcel_to_subcort_pos[parcel_id]
        # Validate index is within matrix bounds
        if idx >= M_subcort.shape[0]:
            raise ValueError(
                f"Computed index {idx} for parcel {parcel_id} is out of bounds "
                f"for matrix size {M_subcort.shape[0]}"
            )
        return idx
    
    # Process by hemisphere first (LH, then RH), then by category
    # Note: Cerebellum (429-430) is NOT in the subcortical matrix (361-424)
    # So we only process subcortical categories here
    category_order = ['Thalamus', 'Basal Ganglia', 'Amygdala']
    
    for hemi in ('LH', 'RH'):
        for category in category_order:
            # Find parcels in this hemisphere and category
            # Only iterate over subcortical region: 361-424 (64 parcels)
            idx = []
            for parcel_id in range(361, 425):  # 361-424 inclusive
                # Skip excluded parcels
                if parcel_id in [390, 423]:
                    continue
                # Ensure parcel is in valid subcortical range
                if parcel_id < 361 or parcel_id > 424:
                    continue
                if parcel_to_category.get(parcel_id) == category:
                    parcel_hemi = get_hemisphere(parcel_id)
                    if parcel_hemi == hemi:
                        # Convert to 0-indexed within subcortical matrix
                        try:
                            subcort_idx = parcel_id_to_subcort_idx(parcel_id)
                            # Validate index is within bounds
                            if subcort_idx >= M_subcort.shape[0]:
                                raise ValueError(
                                    f"Subcortical index {subcort_idx} out of bounds "
                                    f"for matrix size {M_subcort.shape[0]} (parcel {parcel_id})"
                                )
                            idx.append(subcort_idx)
                            assigned_parcels.add(subcort_idx)
                        except ValueError as e:
                            # Skip if parcel is excluded or out of range
                            import warnings
                            warnings.warn(f"Skipping parcel {parcel_id}: {e}")
                            continue
            
            if len(idx) == 0:
                continue
            
            idx = np.array(idx)
            
            # Get submatrix for this block
            Wb = M_subcort[np.ix_(idx, idx)]
            
            # Spectral ordering within block (or use direct index for single-element blocks)
            if len(idx) == 1:
                o = np.array([0])  # Single element, no ordering needed
            else:
                o = _safe_fiedler_order(Wb)
            blocks.append(idx[o])
            block_labels.append(f'{hemi} {category}')
    
    # Note: Brainstem (425-428) and Cerebellum (429-430) are NOT in the subcortical matrix
    # The subcortical matrix only contains parcels 361-424 (64 parcels)
    
    # Add any unassigned parcels to an "Other" category
    # M_subcort has 62 parcels (parcels 361-424 excluding 390 and 423)
    n_subcort_parcels = M_subcort.shape[0]
    all_parcels = set(range(n_subcort_parcels))
    unassigned_list = sorted(list(all_parcels - assigned_parcels))
    
    if len(unassigned_list) > 0:
        unassigned = np.array(unassigned_list)
        # Get submatrix for unassigned parcels
        Wb = M_subcort[np.ix_(unassigned, unassigned)]
        # Spectral ordering
        o = _safe_fiedler_order(Wb)
        blocks.append(unassigned[o])
        block_labels.append('Other')
    
    # Concatenate all blocks
    if blocks:
        order = np.concatenate(blocks)
    else:
        order = np.arange(n_subcort_parcels)
    
    # Validate order
    n_unassigned = len(unassigned_list)
    if len(order) != n_subcort_parcels:
        raise ValueError(f"Expected {n_subcort_parcels} parcels in order, got {len(order)}. "
                        f"Assigned: {len(assigned_parcels)}, Unassigned: {n_unassigned}")
    
    if len(set(order)) != n_subcort_parcels:
        raise ValueError(f"Duplicate parcels in order. Unique count: {len(set(order))}")
    
    if order.min() != 0 or order.max() >= n_subcort_parcels:
        raise ValueError(f"Order indices out of range: min={order.min()}, max={order.max()}, expected 0-{n_subcort_parcels-1}")
    
    # Compute breaks
    breaks = np.cumsum([len(b) for b in blocks])
    
    return order, breaks, block_labels


def build_combined_order_cortical_subcortical(
    M: np.ndarray,
    A7: Union[np.ndarray, Dict[int, int], pd.DataFrame],
    yeo7_names: List[str],
    parcel_to_category: Dict[int, str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build combined ordering for full 430×430 matrix: cortical first, then subcortical.
    
    Parameters:
    -----------
    M : np.ndarray
        Full connectome matrix (430×430)
    A7 : np.ndarray, dict, or pd.DataFrame
        Yeo7 assignment for cortical parcels
    yeo7_names : list
        List of 7 Yeo7 network names
    parcel_to_category : dict
        Maps parcel_id (361-430) to category name
        
    Returns:
    --------
    order : np.ndarray
        0-indexed parcel indices (length 432)
    breaks : np.ndarray
        Block boundaries including cortical and subcortical sections
    block_labels : list
        Block labels for both cortical and subcortical
    """
    if M.shape[0] != 430:
        raise ValueError(f"Expected matrix shape (430, 430), got {M.shape}")
    
    # Get cortical ordering
    cort_order, cort_breaks, cort_labels = build_cortical_order_LH_RH(M, A7, yeo7_names)
    
    # Get subcortical ordering
    subcort_order, subcort_breaks, subcort_labels = build_subcortical_order_by_category(M, parcel_to_category)
    
    # Combine: cortical first (0-359), then subcortical (360-423)
    # For subcortical, we need to map from 0-63 to 360-423
    subcort_order_full = subcort_order + 360
    
    order = np.concatenate([cort_order, subcort_order_full])
    
    # Combine breaks and labels
    breaks = np.concatenate([cort_breaks, subcort_breaks + 360])
    block_labels = cort_labels + subcort_labels
    
    # Validate
    assert len(order) == 430 and len(set(order)) == 430
    assert order.min() == 0 and order.max() == 429
    
    return order, breaks, block_labels


# Color schemes for visualization
YEO7_COLORS = {
    "Visual": "#2E86C1",          # Blue
    "Somatomotor": "#F39C12",     # Orange
    "DorsalAttn": "#27AE60",      # Green
    "VentralAttn": "#C0392B",     # Dark Red
    "Limbic": "#8E44AD",          # Purple
    "FrontoParietal": "#D81B60",  # Pink/Magenta
    "Default": "#E67E22",         # Dark Orange
}

SUBCORTICAL_CATEGORY_COLORS = {
    "Thalamus": "#E74C3C",        # Bright Red (distinct from VentralAttn dark red #C0392B)
    "Basal Ganglia": "#3498DB",   # Light Blue (distinct from Visual blue #2E86C1)
    "Brainstem": "#1ABC9C",       # Turquoise/Cyan (distinct from DorsalAttn green #27AE60)
    "Amygdala": "#9B59B6",        # Violet (distinct from Limbic purple #8E44AD)
    "Cerebellum": "#F1C40F",      # Yellow (distinct from Somatomotor orange #F39C12)
    "Other": "#95A5A6",           # Gray for uncategorized parcels
}


def _compute_connectome_colorscale(M_log: np.ndarray, gamma: float = 0.7) -> Tuple[float, float, PowerNorm]:
    """
    Compute color scale (vmin, vmax) and normalization for connectome visualization.
    
    Parameters:
    -----------
    M_log : np.ndarray
        Log-transformed connectome matrix (e.g., log(1+weight))
    gamma : float
        Gamma parameter for PowerNorm (default: 0.7)
    
    Returns:
    --------
    vmin : float
        Minimum value for color scale
    vmax : float
        Maximum value for color scale
    norm : PowerNorm
        Normalization object for matplotlib
    """
    # Compute color scale from positive values
    pos = M_log[M_log > 0]
    if pos.size:
        vmin, vmax = np.percentile(pos, [2.0, 99.7])
        if vmin >= vmax:
            vmin, vmax = pos.min(), pos.max()
    else:
        vmin, vmax = 0.0, 1.0
    
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    return vmin, vmax, norm


def plot_group_connectome_LH_RH(
    M: np.ndarray,
    A7: Union[np.ndarray, Dict[int, int], pd.DataFrame],
    yeo7_names: List[str],
    title: str = "Group Structural Connectome — Yeo-7 (LH→RH)",
    cmap: str = 'viridis',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12.8, 11.2)
) -> plt.Figure:
    """
    Plot cortical connectome (360×360) with LH→RH ordering by Yeo7 networks.
    
    Parameters:
    -----------
    M : np.ndarray
        Connectome matrix (360×360 or 430×430)
    A7 : np.ndarray, dict, or pd.DataFrame
        Yeo7 assignment for cortical parcels
    yeo7_names : list
        List of 7 Yeo7 network names
    title : str
        Plot title
    cmap : str
        Colormap name (default: 'viridis')
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get ordering
    order, breaks, block_labels = build_cortical_order_LH_RH(M, A7, yeo7_names)
    
    # Reorder matrix
    if M.shape[0] == 430:
        M_cortical = M[:360, :360]
    else:
        M_cortical = M.copy()
    M_ord = M_cortical[np.ix_(order, order)]
    
    # Apply log(1+weight) transformation
    M_log = np.log1p(M_ord)
    
    # Compute color scale using helper function
    vmin, vmax, norm = _compute_connectome_colorscale(M_log, gamma=0.7)
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, width_ratios=[0.8, 11, 1.9], 
                          height_ratios=[0.7, 11, 1.1], 
                          wspace=0.03, hspace=0.08)
    
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.axis('off')
    ax_main = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis('off')
    ax_cbar = fig.add_subplot(gs[2, 1])
    ax_cbar.axis('off')
    
    # Title
    ax_title.text(0.5, 0.5, title, fontsize=18, fontweight='bold', 
                  ha='center', va='center')
    
    # Main matrix
    im = ax_main.imshow(M_log, cmap=cmap, norm=norm, aspect='equal',
                        extent=[-0.5, M_log.shape[1] - 0.5, M_log.shape[0] - 0.5, -0.5])
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    # Explicitly set axis limits to ensure matrix fills entire space
    ax_main.set_xlim(-0.5, M_log.shape[1] - 0.5)
    ax_main.set_ylim(M_log.shape[0] - 0.5, -0.5)
    
    # Calculate block start and end positions
    starts = np.r_[0, breaks[:-1]]
    ends = breaks
    
    # Draw top and right borders only (line segments, not full grid)
    for s, e in zip(starts, ends):
        # Top border: horizontal line segment
        ax_main.plot([s, e], [s - 0.5, s - 0.5], color='w', lw=1.0, alpha=0.95)
        # Right border: vertical line segment
        ax_main.plot([e - 0.5, e - 0.5], [s, e], color='w', lw=1.0, alpha=0.95)
    
    # Note: LH/RH separator lines removed per user request to avoid thick unwanted lines
    # The block borders (lw=1.0) provide sufficient visual separation
    
    # Top assignment bar
    topbar = fig.add_axes([
        ax_main.get_position().x0,
        ax_main.get_position().y1 + 0.006,
        ax_main.get_position().width,
        0.015
    ])
    topbar.set_xlim(0, len(order))
    topbar.set_ylim(0, 1)
    topbar.axis('off')
    
    for (s, e), lbl in zip(zip(starts, ends), block_labels):
        net = lbl.split(' ', 1)[1]  # Remove 'LH ' or 'RH '
        topbar.add_patch(Rectangle((s, 0), e - s, 1, 
                                  color=YEO7_COLORS.get(net, '#9e9e9e'), 
                                  ec='none'))
    
    # Note: LH/RH split lines removed from top and left bars per user request
    
    # Left assignment bar
    leftbar = fig.add_axes([
        ax_main.get_position().x0 - 0.02,
        ax_main.get_position().y0,
        0.015,
        ax_main.get_position().height
    ])
    leftbar.set_ylim(len(order), 0)  # Invert y-axis
    leftbar.set_xlim(0, 1)
    leftbar.axis('off')
    
    for (s, e), lbl in zip(zip(starts, ends), block_labels):
        net = lbl.split(' ', 1)[1]
        leftbar.add_patch(Rectangle((0, s), 1, e - s, 
                                   color=YEO7_COLORS.get(net, '#9e9e9e'), 
                                   ec='none'))
    
    # Note: LH/RH split lines removed from top and left bars per user request
    
    # Legend
    y = 0.95
    for name in yeo7_names:
        col = YEO7_COLORS.get(name, '#9e9e9e')
        ax_leg.add_patch(Rectangle((0.08, y - 0.03), 0.09, 0.06, 
                                   color=col, ec='k', lw=0.5))
        ax_leg.text(0.20, y, name, fontsize=12, va='center')
        y -= 0.085
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_cbar, orientation='horizontal', 
                       fraction=0.78, pad=0.25)
    cbar.set_label('log(1+weight)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_subcortical_connectome(
    M: np.ndarray,
    parcel_to_category: Dict[int, str],
    title: str = "Group Structural Connectome — Subcortical",
    cmap: str = 'viridis',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 9)
) -> plt.Figure:
    """
    Plot subcortical connectome (72×72) with category-based ordering.
    
    Parameters:
    -----------
    M : np.ndarray
        Connectome matrix (64×64 or 430×430)
    parcel_to_category : dict
        Maps parcel_id (361-430) to category name
    title : str
        Plot title
    cmap : str
        Colormap name (default: 'viridis')
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get ordering
    order, breaks, block_labels = build_subcortical_order_by_category(M, parcel_to_category)
    
    # Reorder matrix
    if M.shape[0] == 430:
        M_subcort = M[360:424, 360:424]
    else:
        M_subcort = M.copy()
    M_ord = M_subcort[np.ix_(order, order)]
    
    # Apply log(1+weight) transformation
    M_log = np.log1p(M_ord)
    
    # Compute color scale using helper function
    vmin, vmax, norm = _compute_connectome_colorscale(M_log, gamma=0.7)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, width_ratios=[0.8, 10, 1.5], 
                          height_ratios=[0.7, 10, 1.1], 
                          wspace=0.03, hspace=0.08)
    
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.axis('off')
    ax_main = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis('off')
    ax_cbar = fig.add_subplot(gs[2, 1])
    ax_cbar.axis('off')
    
    # Title
    ax_title.text(0.5, 0.5, title, fontsize=18, fontweight='bold', 
                  ha='center', va='center')
    
    # Main matrix
    im = ax_main.imshow(M_log, cmap=cmap, norm=norm, aspect='equal',
                        extent=[-0.5, M_log.shape[1] - 0.5, M_log.shape[0] - 0.5, -0.5])
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    # Explicitly set axis limits to ensure matrix fills entire space
    ax_main.set_xlim(-0.5, M_log.shape[1] - 0.5)
    ax_main.set_ylim(M_log.shape[0] - 0.5, -0.5)
    
    # Calculate block positions
    starts = np.r_[0, breaks[:-1]]
    ends = breaks
    
    # Draw top and right borders only
    for s, e in zip(starts, ends):
        # Top border
        ax_main.plot([s, e], [s - 0.5, s - 0.5], color='w', lw=1.0, alpha=0.95)
        # Right border
        ax_main.plot([e - 0.5, e - 0.5], [s, e], color='w', lw=1.0, alpha=0.95)
    
    # Note: LH/RH separator lines removed per user request to avoid thick unwanted lines
    # The block borders (lw=1.0) provide sufficient visual separation
    
    # Top assignment bar
    topbar = fig.add_axes([
        ax_main.get_position().x0,
        ax_main.get_position().y1 + 0.006,
        ax_main.get_position().width,
        0.015
    ])
    topbar.set_xlim(0, len(order))
    topbar.set_ylim(0, 1)
    topbar.axis('off')
    
    # Helper function to extract category name from label (handles "LH Thalamus" -> "Thalamus")
    def get_category_from_label(lbl):
        """Extract category name from label, handling LH/RH prefixes"""
        if lbl.startswith(('LH ', 'RH ')):
            return lbl.split(' ', 1)[1]
        return lbl
    
    for (s, e), lbl in zip(zip(starts, ends), block_labels):
        category = get_category_from_label(lbl)
        topbar.add_patch(Rectangle((s, 0), e - s, 1, 
                                  color=SUBCORTICAL_CATEGORY_COLORS.get(category, '#9e9e9e'), 
                                  ec='none'))
    
    # Note: LH/RH split lines removed from top and left bars per user request
    
    # Left assignment bar
    leftbar = fig.add_axes([
        ax_main.get_position().x0 - 0.02,
        ax_main.get_position().y0,
        0.015,
        ax_main.get_position().height
    ])
    leftbar.set_ylim(len(order), 0)
    leftbar.set_xlim(0, 1)
    leftbar.axis('off')
    
    for (s, e), lbl in zip(zip(starts, ends), block_labels):
        category = get_category_from_label(lbl)
        leftbar.add_patch(Rectangle((0, s), 1, e - s, 
                                   color=SUBCORTICAL_CATEGORY_COLORS.get(category, '#9e9e9e'), 
                                   ec='none'))
    
    # Note: LH/RH split lines removed from top and left bars per user request
    
    # Legend - always show all categories (even if they don't appear in block_labels)
    y = 0.95
    all_categories = ['Thalamus', 'Basal Ganglia', 'Brainstem', 'Amygdala', 'Cerebellum', 'Other']
    # Get unique categories that appear in block_labels (extract from LH/RH labels)
    categories_in_labels = set([get_category_from_label(lbl) for lbl in block_labels])
    for category in all_categories:
        # Always show category in legend, even if not in current block_labels
        col = SUBCORTICAL_CATEGORY_COLORS.get(category, '#9e9e9e')
        ax_leg.add_patch(Rectangle((0.08, y - 0.03), 0.09, 0.06, 
                                   color=col, ec='k', lw=0.5))
        ax_leg.text(0.20, y, category, fontsize=12, va='center')
        y -= 0.085
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_cbar, orientation='horizontal', 
                       fraction=0.78, pad=0.25)
    cbar.set_label('log(1+weight)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



def plot_combined_connectome_cortical_subcortical(
    M: np.ndarray,
    A7: Union[np.ndarray, Dict[int, int], pd.DataFrame],
    yeo7_names: List[str],
    parcel_to_category: Dict[int, str],
    title: str = "Group Structural Connectome — Cortical + Subcortical",
    cmap: str = 'viridis',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (13, 12)
) -> plt.Figure:
    """
    Plot full 430×430 connectome with cortical first, then subcortical.
    
    Parameters:
    -----------
    M : np.ndarray
        Full connectome matrix (430×430)
    A7 : np.ndarray, dict, or pd.DataFrame
        Yeo7 assignment for cortical parcels
    yeo7_names : list
        List of 7 Yeo7 network names
    parcel_to_category : dict
        Maps parcel_id (361-430) to category name
    title : str
        Plot title
    cmap : str
        Colormap name (default: 'viridis')
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if M.shape[0] != 430:
        raise ValueError(f"Expected matrix shape (430, 430), got {M.shape}")
    
    # Get combined ordering
    order, breaks, block_labels = build_combined_order_cortical_subcortical(
        M, A7, yeo7_names, parcel_to_category
    )
    
    # Reorder matrix
    M_ord = M[np.ix_(order, order)]
    
    # Apply log(1+weight) transformation
    M_log = np.log1p(M_ord)
    
    # Compute color scale using helper function
    vmin, vmax, norm = _compute_connectome_colorscale(M_log, gamma=0.7)
    
    # Find cortical/subcortical split (first subcortical block)
    subcort_start_idx = None
    for i, lbl in enumerate(block_labels):
        if not lbl.startswith(('LH', 'RH')):
            subcort_start_idx = i
            break
    if subcort_start_idx is not None and subcort_start_idx > 0:
        subcort_start = breaks[subcort_start_idx - 1]
    else:
        subcort_start = 360
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, width_ratios=[0.8, 11, 2.0], 
                          height_ratios=[0.7, 11, 1.1], 
                          wspace=0.03, hspace=0.08)
    
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.axis('off')
    ax_main = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis('off')
    ax_cbar = fig.add_subplot(gs[2, 1])
    ax_cbar.axis('off')
    
    # Title
    ax_title.text(0.5, 0.5, title, fontsize=18, fontweight='bold', 
                  ha='center', va='center')
    
    # Main matrix
    im = ax_main.imshow(M_log, cmap=cmap, norm=norm, aspect='equal',
                        extent=[-0.5, M_log.shape[1] - 0.5, M_log.shape[0] - 0.5, -0.5])
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    # Explicitly set axis limits to ensure matrix fills entire space
    ax_main.set_xlim(-0.5, M_log.shape[1] - 0.5)
    ax_main.set_ylim(M_log.shape[0] - 0.5, -0.5)
    
    # Calculate block positions
    starts = np.r_[0, breaks[:-1]]
    ends = breaks
    
    # Draw top and right borders only
    for s, e in zip(starts, ends):
        # Top border
        ax_main.plot([s, e], [s - 0.5, s - 0.5], color='w', lw=1.0, alpha=0.95)
        # Right border
        ax_main.plot([e - 0.5, e - 0.5], [s, e], color='w', lw=1.0, alpha=0.95)
    
    # Bold cortical/subcortical separator
    ax_main.axhline(subcort_start - 0.5, color='w', lw=2.5, alpha=1.0)
    ax_main.axvline(subcort_start - 0.5, color='w', lw=2.5, alpha=1.0)
    
    # Note: LH/RH separator lines removed per user request to avoid thick unwanted lines
    # The block borders (lw=1.0) and cortical/subcortical separator provide sufficient visual separation
    
    # Top assignment bar
    topbar = fig.add_axes([
        ax_main.get_position().x0,
        ax_main.get_position().y1 + 0.006,
        ax_main.get_position().width,
        0.015
    ])
    topbar.set_xlim(0, len(order))
    topbar.set_ylim(0, 1)
    topbar.axis('off')
    
    # Helper function to extract category name from label
    def get_category_from_label_combined(lbl):
        """Extract category name from label, handling LH/RH prefixes for both cortical and subcortical"""
        if lbl.startswith(('LH ', 'RH ')):
            return lbl.split(' ', 1)[1]
        return lbl
    
    for (s, e), lbl in zip(zip(starts, ends), block_labels):
        if lbl.startswith(('LH', 'RH')):
            # Check if it's cortical (Yeo7 network) or subcortical
            parts = lbl.split(' ', 1)
            if len(parts) > 1:
                name = parts[1]
                # Check if it's a Yeo7 network name
                if name in yeo7_names:
                    col = YEO7_COLORS.get(name, '#9e9e9e')
                else:
                    # Subcortical category
                    col = SUBCORTICAL_CATEGORY_COLORS.get(name, '#9e9e9e')
            else:
                col = '#9e9e9e'
        else:
            col = SUBCORTICAL_CATEGORY_COLORS.get(lbl, '#9e9e9e')
        topbar.add_patch(Rectangle((s, 0), e - s, 1, color=col, ec='none'))
    
    # Cortical/subcortical separator on top bar
    topbar.plot([subcort_start, subcort_start], [0, 1], color='w', lw=2.5, alpha=1.0)
    
    # Left assignment bar
    leftbar = fig.add_axes([
        ax_main.get_position().x0 - 0.02,
        ax_main.get_position().y0,
        0.015,
        ax_main.get_position().height
    ])
    leftbar.set_ylim(len(order), 0)
    leftbar.set_xlim(0, 1)
    leftbar.axis('off')
    
    for (s, e), lbl in zip(zip(starts, ends), block_labels):
        if lbl.startswith(('LH', 'RH')):
            # Check if it's cortical (Yeo7 network) or subcortical
            parts = lbl.split(' ', 1)
            if len(parts) > 1:
                name = parts[1]
                # Check if it's a Yeo7 network name
                if name in yeo7_names:
                    col = YEO7_COLORS.get(name, '#9e9e9e')
                else:
                    # Subcortical category
                    col = SUBCORTICAL_CATEGORY_COLORS.get(name, '#9e9e9e')
            else:
                col = '#9e9e9e'
        else:
            col = SUBCORTICAL_CATEGORY_COLORS.get(lbl, '#9e9e9e')
        leftbar.add_patch(Rectangle((0, s), 1, e - s, color=col, ec='none'))
    
    # Cortical/subcortical separator on left bar
    leftbar.plot([0, 1], [subcort_start, subcort_start], color='w', lw=2.5, alpha=1.0)
    
    # Combined legend
    y = 0.95
    ax_leg.text(0.1, y, "Cortical Networks:", fontsize=12, fontweight='bold')
    y -= 0.08
    for name in yeo7_names:
        col = YEO7_COLORS.get(name, '#9e9e9e')
        ax_leg.add_patch(Rectangle((0.08, y - 0.03), 0.09, 0.06, 
                                   color=col, ec='k', lw=0.5))
        ax_leg.text(0.20, y, name, fontsize=11, va='center')
        y -= 0.07
    
    y -= 0.05
    ax_leg.text(0.1, y, "Subcortical:", fontsize=12, fontweight='bold')
    y -= 0.08
    # Always show all subcortical categories, even if they don't appear in block_labels
    for category in ['Thalamus', 'Basal Ganglia', 'Brainstem', 'Amygdala', 'Cerebellum', 'Other']:
        col = SUBCORTICAL_CATEGORY_COLORS.get(category, '#9e9e9e')
        ax_leg.add_patch(Rectangle((0.08, y - 0.03), 0.09, 0.06, 
                                   color=col, ec='k', lw=0.5))
        ax_leg.text(0.20, y, category, fontsize=11, va='center')
        y -= 0.07
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_cbar, orientation='horizontal', 
                       fraction=0.78, pad=0.25)
    cbar.set_label('log(1+weight)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multi_group_connectomes(
    connectomes_dict: Dict[str, np.ndarray],
    groups: Dict[str, List[str]],
    A7: Union[np.ndarray, Dict[int, int], pd.DataFrame],
    yeo7_names: List[str],
    parcel_to_category: Dict[int, str],
    plot_type: str = 'combined',
    connectome_type: str = 'SC_sift2_sizecorr',
    cmap: str = 'viridis',
    save_dir: Optional[Union[str, Path]] = None
) -> Dict[str, plt.Figure]:
    """
    Create multi-group connectome visualizations.
    
    Parameters:
    -----------
    connectomes_dict : dict
        Dictionary mapping subject IDs to connectome matrices
    groups : dict
        Dictionary mapping group names to subject ID lists
    A7 : np.ndarray, dict, or pd.DataFrame
        Yeo7 assignment for cortical parcels
    yeo7_names : list
        List of 7 Yeo7 network names
    parcel_to_category : dict
        Maps parcel_id (361-430) to category name
    plot_type : str
        Type of plot: 'cortical', 'subcortical', 'combined', or 'all'
    connectome_type : str
        Connectome type name (for file naming)
    cmap : str
        Colormap name (default: 'viridis')
    save_dir : str or Path, optional
        Directory to save figures
        
    Returns:
    --------
    figures : dict
        Dictionary mapping group names to figure objects
    """
    save_dir = Path(save_dir) if save_dir else None
    
    # Compute group averages
    group_averages = {}
    for group_name, subjects in groups.items():
        group_connectomes = {}
        for subj in subjects:
            subj_key = f'sub-{subj}' if not subj.startswith('sub-') else subj
            if subj_key in connectomes_dict:
                group_connectomes[subj_key] = connectomes_dict[subj_key]
        
        if len(group_connectomes) > 0:
            group_averages[group_name] = compute_group_average(group_connectomes)
    
    figures = {}
    
    if plot_type == 'all':
        # Create separate figures for each group and each plot type
        for group_name, M_avg in group_averages.items():
            # Cortical
            fig = plot_group_connectome_LH_RH(
                M_avg, A7, yeo7_names,
                title=f"{group_name} — Cortical Connectome",
                cmap=cmap,
                save_path=save_dir / f'{connectome_type}_{group_name.replace(" ", "_")}_cortical.png' if save_dir else None
            )
            figures[f'{group_name}_cortical'] = fig
            
            # Subcortical
            fig = plot_subcortical_connectome(
                M_avg, parcel_to_category,
                title=f"{group_name} — Subcortical Connectome",
                cmap=cmap,
                save_path=save_dir / f'{connectome_type}_{group_name.replace(" ", "_")}_subcortical.png' if save_dir else None
            )
            figures[f'{group_name}_subcortical'] = fig
            
            # Combined
            fig = plot_combined_connectome_cortical_subcortical(
                M_avg, A7, yeo7_names, parcel_to_category,
                title=f"{group_name} — Combined Connectome",
                cmap=cmap,
                save_path=save_dir / f'{connectome_type}_{group_name.replace(" ", "_")}_combined.png' if save_dir else None
            )
            figures[f'{group_name}_combined'] = fig
    else:
        # Create individual plots for each group
        for group_name, M_avg in group_averages.items():
            if plot_type == 'cortical':
                fig = plot_group_connectome_LH_RH(
                    M_avg, A7, yeo7_names,
                    title=f"{group_name}",
                    cmap=cmap,
                    save_path=save_dir / f'{connectome_type}_{group_name.replace(" ", "_")}_cortical.png' if save_dir else None
                )
                figures[f'{group_name}_cortical'] = fig
            elif plot_type == 'subcortical':
                fig = plot_subcortical_connectome(
                    M_avg, parcel_to_category,
                    title=f"{group_name}",
                    cmap=cmap,
                    save_path=save_dir / f'{connectome_type}_{group_name.replace(" ", "_")}_subcortical.png' if save_dir else None
                )
                figures[f'{group_name}_subcortical'] = fig
            elif plot_type == 'combined':
                fig = plot_combined_connectome_cortical_subcortical(
                    M_avg, A7, yeo7_names, parcel_to_category,
                    title=f"{group_name}",
                    cmap=cmap,
                    save_path=save_dir / f'{connectome_type}_{group_name.replace(" ", "_")}_combined.png' if save_dir else None
                )
                figures[f'{group_name}_combined'] = fig
    
    return figures
