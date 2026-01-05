"""
Visualization functions for SDI analysis.

All figures use the standard color scheme:
- ICU: #E67E22 (orange)
- ICU Delirium: #C0392B (dark red/orange)
- Healthy Controls: #3498DB (blue)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from scipy import stats

logger = logging.getLogger(__name__)

# Import brain projection function
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from plot.brain_projection import brain_projection
except ImportError:
    brain_projection = None
    logger.warning("Could not import brain_projection function")

try:
    from utils.group import get_group_color
except ImportError:
    get_group_color = None

# Color scheme
GROUP_COLORS = {
    'ICU': '#E67E22',
    'ICU Delirium': '#C0392B',
    'Healthy Controls': '#3498DB'
}


def plot_distributions(
    unified_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    use_log_scale: bool = True,
    percentile_clip: Tuple[float, float] = (1, 99),
    combine_plots: bool = True
) -> plt.Figure:
    """
    Create combined violin and boxplot for group-level SDI distributions.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    use_log_scale : bool
        Whether to use log scale for y-axis (default: True)
    percentile_clip : tuple
        Percentiles to use for y-axis limits (default: (1, 99))
    combine_plots : bool
        Combine violin and boxplot in single plot (default: True)
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    if combine_plots:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    groups = sorted(unified_df['group'].unique())
    colors = [GROUP_COLORS.get(g, '#95A5A6') for g in groups]
    
    # Prepare data - clean NaN/inf values for each group
    data_to_plot = []
    for g in groups:
        group_data = unified_df[unified_df['group'] == g]['sdi_value'].values
        # Clean data: remove NaN, inf, and non-positive values
        group_data_clean = group_data[np.isfinite(group_data) & (group_data > 0)]
        if len(group_data_clean) == 0:
            logger.warning(f"No valid data for group {g}")
        data_to_plot.append(group_data_clean if len(group_data_clean) > 0 else group_data)
    
    # Calculate y-axis limits based on percentiles (excluding extreme outliers)
    all_data = np.concatenate([d for d in data_to_plot if len(d) > 0])
    all_data_clean = all_data[np.isfinite(all_data) & (all_data > 0)]
    
    if len(all_data_clean) > 0:
        if use_log_scale:
            # For log scale, use percentiles of log-transformed data
            log_data = np.log2(all_data_clean + 1e-10)
            y_min = np.percentile(log_data, percentile_clip[0])
            y_max = np.percentile(log_data, percentile_clip[1])
            y_label = 'SDI Value (log2)'
            # Transform each group's data, handling empty arrays
            data_to_plot_transformed = []
            for d in data_to_plot:
                if len(d) > 0:
                    d_clean = d[np.isfinite(d) & (d > 0)]
                    if len(d_clean) > 0:
                        data_to_plot_transformed.append(np.log2(d_clean + 1e-10))
                    else:
                        data_to_plot_transformed.append(np.array([]))
                else:
                    data_to_plot_transformed.append(np.array([]))
        else:
            # For linear scale, use percentiles of raw data
            y_min = np.percentile(all_data_clean, percentile_clip[0])
            y_max = np.percentile(all_data_clean, percentile_clip[1])
            y_label = 'SDI Value'
            # Add some padding
            y_range = y_max - y_min
            y_min = max(0, y_min - 0.05 * y_range)
            y_max = y_max + 0.05 * y_range
            # Clean each group's data
            data_to_plot_transformed = []
            for d in data_to_plot:
                d_clean = d[np.isfinite(d) & (d > 0)]
                data_to_plot_transformed.append(d_clean if len(d_clean) > 0 else np.array([]))
    else:
        y_min, y_max = 0, 1
        y_label = 'SDI Value'
        data_to_plot_transformed = data_to_plot
    
    # Calculate summary statistics for annotation
    summary_stats = []
    for g, data in zip(groups, data_to_plot):
        data_clean = data[np.isfinite(data) & (data > 0)]
        if len(data_clean) > 0:
            median_val = np.median(data_clean)
            q25 = np.percentile(data_clean, 25)
            q75 = np.percentile(data_clean, 75)
            iqr = q75 - q25
            summary_stats.append(f"{g}: median={median_val:.3f}, IQR=[{q25:.3f}, {q75:.3f}]")
    
    # Create combined plot
    if combine_plots:
        ax = axes[0]
        positions = np.arange(len(groups))
        
        # Check if we have any valid data
        valid_data_count = sum(1 for d in data_to_plot_transformed if len(d) > 0)
        if valid_data_count == 0:
            logger.error("No valid data to plot")
            ax.text(0.5, 0.5, 'No valid data to plot', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            return fig
        
        # Prepare data for violin plot - ensure all arrays have at least one value
        # (violinplot requires non-empty arrays, but we'll use a dummy value for empty ones)
        violin_data = []
        for d in data_to_plot_transformed:
            if len(d) > 0:
                violin_data.append(d)
            else:
                # Use a single dummy value far outside the plot range
                # This ensures violinplot doesn't error, but won't be visible
                dummy_val = y_min - 10 * (y_max - y_min) if y_max > y_min else -1
                violin_data.append(np.array([dummy_val]))
        
        # Create violin plot with better visibility settings
        # Use same positions as boxplot to ensure alignment
        try:
            parts = ax.violinplot(violin_data, positions=positions, 
                                 showmeans=False, showmedians=False, showextrema=True, 
                                 widths=0.6)
            
            # Color violins with higher alpha for better visibility
            # Only style violins that have valid data
            for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
                if len(data_to_plot_transformed[i]) > 0:  # Only style if has data
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)  # Increased from 0.5 for better visibility
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)
        except Exception as e:
            logger.warning(f"Error creating violin plot: {e}. Continuing with boxplot only.")
            parts = None
        
        # Overlay boxplot on all groups (boxplot handles empty arrays gracefully)
        bp = ax.boxplot(data_to_plot_transformed, positions=positions, 
                       widths=0.15, patch_artist=True, showfliers=False,
                       boxprops=dict(alpha=0.9, linewidth=1.5),
                       medianprops=dict(color='black', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        
        # Set axis properties BEFORE setting limits to avoid clipping violins
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title('SDI Distribution (Violin + Boxplot)', fontsize=14, fontweight='bold')
        
        # Set y-axis scale BEFORE setting limits
        if use_log_scale:
            ax.set_yscale('log')
        
        # Set y-axis limits with a bit more padding to ensure violins aren't clipped
        # Calculate actual data range including violins
        all_plot_data = np.concatenate([d for d in data_to_plot_transformed if len(d) > 0])
        if len(all_plot_data) > 0:
            data_min = np.min(all_plot_data)
            data_max = np.max(all_plot_data)
            data_range = data_max - data_min
            # Use the wider of percentile-based or data-based range, with padding
            y_min_final = min(y_min, data_min - 0.1 * data_range)
            y_max_final = max(y_max, data_max + 0.1 * data_range)
        else:
            y_min_final, y_max_final = y_min, y_max
        
        ax.set_ylim(y_min_final, y_max_final)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size and summary statistics
        n_parcels = len(unified_df[unified_df['group'] == groups[0]])
        ax.text(0.02, 0.98, f'n = {n_parcels} parcels per group', 
               transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add summary statistics as text
        stats_text = '\n'.join(summary_stats[:3])  # Show first 3 groups
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        # Separate plots (original behavior)
        # Violin plot
        parts = axes[0].violinplot(data_to_plot_transformed, positions=range(len(groups)), 
                                   showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        axes[0].set_xticks(range(len(groups)))
        axes[0].set_xticklabels(groups, rotation=45, ha='right')
        axes[0].set_ylabel(y_label, fontsize=12)
        axes[0].set_title('SDI Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        axes[0].set_ylim(y_min, y_max)
        if use_log_scale:
            axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Boxplot
        bp = axes[1].boxplot(data_to_plot_transformed, labels=groups, patch_artist=True, 
                            showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel(y_label, fontsize=12)
        axes[1].set_title('SDI Distribution (Boxplot)', fontsize=14, fontweight='bold')
        axes[1].set_ylim(y_min, y_max)
        if use_log_scale:
            axes[1].set_yscale('log')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution plots to {output_path}")
    
    return fig


def plot_group_comparison(
    unified_df: pd.DataFrame,
    posthoc_results: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    use_log_scale: bool = True,
    percentile_clip: Tuple[float, float] = (1, 99)
) -> plt.Figure:
    """
    Create boxplot with statistical annotations.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    posthoc_results : pd.DataFrame, optional
        Post-hoc test results for annotations
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    use_log_scale : bool
        Whether to use log scale for y-axis (default: True)
    percentile_clip : tuple
        Percentiles to use for y-axis limits (default: (1, 99))
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = sorted(unified_df['group'].unique())
    colors = [GROUP_COLORS.get(g, '#95A5A6') for g in groups]
    
    data_to_plot = [unified_df[unified_df['group'] == g]['sdi_value'].values 
                    for g in groups]
    
    # Calculate y-axis limits based on percentiles (excluding extreme outliers)
    all_data = np.concatenate(data_to_plot)
    all_data_clean = all_data[np.isfinite(all_data) & (all_data > 0)]
    
    if len(all_data_clean) > 0:
        if use_log_scale:
            # For log scale, use percentiles of log-transformed data
            log_data = np.log2(all_data_clean + 1e-10)
            y_min = np.percentile(log_data, percentile_clip[0])
            y_max = np.percentile(log_data, percentile_clip[1])
            y_label = 'SDI Value (log2)'
            # Transform data for plotting
            data_to_plot_transformed = [np.log2(d + 1e-10) for d in data_to_plot]
        else:
            # For linear scale, use percentiles of raw data
            y_min = np.percentile(all_data_clean, percentile_clip[0])
            y_max = np.percentile(all_data_clean, percentile_clip[1])
            y_label = 'SDI Value'
            # Add some padding
            y_range = y_max - y_min
            y_min = max(0, y_min - 0.05 * y_range)
            y_max = y_max + 0.05 * y_range
            data_to_plot_transformed = data_to_plot
    else:
        y_min, y_max = 0, 1
        y_label = 'SDI Value'
        data_to_plot_transformed = data_to_plot
    
    # Create boxplot (hide outliers for cleaner view, focus on main distribution)
    bp = ax.boxplot(data_to_plot_transformed, labels=groups, patch_artist=True, 
                    widths=0.6, showmeans=True, showfliers=False)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set y-axis limits and scale
    ax.set_ylim(y_min, y_max)
    if use_log_scale:
        ax.set_yscale('log')
    
    
    
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_xlabel('Group', fontsize=15)
    ax.set_title('Group Comparison of SDI Values', fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved group comparison plot to {output_path}")
    
    return fig


def plot_brain_projections(
    group_arrays: Dict[str, np.ndarray],
    config: Dict,
    output_dir: str,
    prefix: str = "sdi",
    use_consistent_scale: bool = True,
    use_log2: bool = True,
    use_median: bool = False
) -> Dict[str, str]:
    """
    Create brain projection maps for each group with consistent color scaling.
    
    Parameters:
    -----------
    group_arrays : dict
        Dictionary mapping group names to SDI arrays.
        Can be (430,) for group-averaged data or (n_subjects, 430) for individual data.
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory
    prefix : str
        File prefix
    use_consistent_scale : bool
        Use consistent color scale across all groups (default: True)
    use_log2 : bool
        Use log2 transformation (default: True)
    use_median : bool
        If individual data (2D array), use median instead of mean (default: False)
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping group names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Process arrays: if 2D (individual data), compute group average
    processed_arrays = {}
    for group_name, sdi_array in group_arrays.items():
        if sdi_array.ndim == 2:
            # Individual data: (n_subjects, 430) -> compute mean/median across subjects
            if use_median:
                processed_arrays[group_name] = np.median(sdi_array, axis=0)
            else:
                processed_arrays[group_name] = np.mean(sdi_array, axis=0)
            logger.debug(f"Group {group_name}: computed {'median' if use_median else 'mean'} from {sdi_array.shape[0]} subjects")
        elif sdi_array.ndim == 1:
            # Group-averaged data: (430,)
            processed_arrays[group_name] = sdi_array
        else:
            raise ValueError(f"Unexpected array shape for {group_name}: {sdi_array.shape}")
    
    # Calculate global min/max for consistent scaling
    if use_consistent_scale:
        all_values = np.concatenate(list(processed_arrays.values()))
        if use_log2:
            all_values_transformed = np.log2(all_values + 1e-10)
        else:
            all_values_transformed = all_values
        global_min = np.nanmin(all_values_transformed)
        global_max = np.nanmax(all_values_transformed)
        logger.info(f"Global value range for consistent scaling: [{global_min:.3f}, {global_max:.3f}]")
    else:
        global_min, global_max = None, None
    
    for group_name, sdi_array in processed_arrays.items():
        # Transform values
        if use_log2:
            transformed_sdi = np.log2(sdi_array + 1e-10)
            cbar_label = "SDI (log2)"
            # Calculate actual range for this group
            vmin, vmax = np.nanmin(transformed_sdi), np.nanmax(transformed_sdi)
            cbar_label += f"\n(range: {vmin:.2f} to {vmax:.2f})"
        else:
            transformed_sdi = sdi_array
            cbar_label = "SDI Value"
            vmin, vmax = np.nanmin(transformed_sdi), np.nanmax(transformed_sdi)
            cbar_label += f"\n(range: {vmin:.2e} to {vmax:.2e})"
        
        save_path = output_path / f"{prefix}_brain_projection_{group_name.replace(' ', '_')}.png"
        
        try:
            if brain_projection is None:
                logger.warning(f"brain_projection function not available, skipping {group_name}")
                file_paths[group_name] = None
            else:
                brain_projection(
                    config,
                    transformed_sdi,
                    str(save_path),
                    cbar_label=cbar_label,
                    # cbar_endlabels=("Low SDI", "High SDI")
                    cbar_endlabels=("Coupling", "Decoupling")
                )
                file_paths[group_name] = str(save_path)
                logger.info(f"Saved brain projection for {group_name} to {save_path}")
        except Exception as e:
            logger.error(f"Error creating brain projection for {group_name}: {e}")
            file_paths[group_name] = None
    
    return file_paths


def plot_brain_projections_sidebyside(
    group_arrays: Dict[str, np.ndarray],
    config: Dict,
    output_path: str,
    prefix: str = "sdi",
    use_log2: bool = True
) -> str:
    """
    Create side-by-side comparison of brain projections for all groups.
    
    Parameters:
    -----------
    group_arrays : dict
        Dictionary mapping group names to SDI arrays (430,)
    config : dict
        Configuration dictionary
    output_path : str
        Output file path
    prefix : str
        File prefix
    use_log2 : bool
        Use log2 transformation (default: True)
    
    Returns:
    --------
    file_path : str
        Path to saved figure
    """
    # Note: This would require modifying brain_projection to support subplots
    # For now, this is a placeholder that creates individual plots
    # Full implementation would require access to brain_projection internals
    logger.warning("Side-by-side brain projections not yet fully implemented. "
                  "Creating individual plots instead.")
    
    output_dir = str(Path(output_path).parent)
    file_paths = plot_brain_projections(
        group_arrays, config, output_dir, prefix=prefix, 
        use_consistent_scale=True, use_log2=use_log2
    )
    
    return output_path


def plot_difference_maps(
    group_arrays: Dict[str, np.ndarray],
    config: Dict,
    output_dir: str,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    prefix: str = "sdi",
    significance_maps: Optional[Dict[str, np.ndarray]] = None,
    use_log_transform: bool = True,
    effect_size_threshold: Optional[float] = None,
    use_percent_change: bool = False,
    use_median: bool = False
) -> Dict[str, str]:
    """
    Create brain projection maps showing group differences with statistical overlay.
    
    Parameters:
    -----------
    group_arrays : dict
        Dictionary mapping group names to SDI arrays.
        Can be (430,) for group-averaged data or (n_subjects, 430) for individual data.
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory
    comparisons : list, optional
        List of (group1, group2) tuples. If None, compares all pairs.
    prefix : str
        File prefix
    significance_maps : dict, optional
        Dictionary mapping comparison names to significance maps (binary arrays)
    use_log_transform : bool
        Use log-transformed differences (default: True)
    effect_size_threshold : float, optional
        Only show differences above this threshold (in original units)
    use_percent_change : bool
        Show percent change instead of absolute difference (default: False)
    use_median : bool
        If individual data (2D array), use median instead of mean (default: False)
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping comparison names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if comparisons is None:
        groups = sorted(group_arrays.keys())
        comparisons = [(g1, g2) for i, g1 in enumerate(groups) 
                      for g2 in groups[i+1:]]
    
    file_paths = {}
    
    for group1, group2 in comparisons:
        if group1 not in group_arrays or group2 not in group_arrays:
            continue
        
        comp_name = f"{group1.replace(' ', '_')}_vs_{group2.replace(' ', '_')}"
        
        # Get arrays and compute group averages if needed
        arr1, arr2 = group_arrays[group1], group_arrays[group2]
        
        # Handle 2D arrays (individual data)
        if arr1.ndim == 2:
            arr1 = np.median(arr1, axis=0) if use_median else np.mean(arr1, axis=0)
        if arr2.ndim == 2:
            arr2 = np.median(arr2, axis=0) if use_median else np.mean(arr2, axis=0)
        
        # Calculate difference
        
        if use_percent_change:
            # Percent change: ((group1 - group2) / group2) * 100
            diff = ((arr1 - arr2) / (arr2 + 1e-10)) * 100
            cbar_label = f"SDI Percent Change ({group1} - {group2}) / {group2} × 100"
        else:
            diff = arr1 - arr2
            cbar_label = f"SDI Difference ({group1} - {group2})"
        
        # Apply effect size threshold if specified
        if effect_size_threshold is not None:
            mask = np.abs(diff) < effect_size_threshold
            diff[mask] = 0
        
        # Apply significance overlay if available
        if significance_maps is not None and comp_name in significance_maps:
            sig_map = significance_maps[comp_name]
            # Mask non-significant differences
            diff[~sig_map.astype(bool)] = 0
            cbar_label += " (significant only)"
        
        # Apply log transformation if requested (for absolute differences only)
        if use_log_transform and not use_percent_change:
            # Use signed log transform: sign(diff) * log2(1 + |diff|)
            diff_transformed = np.sign(diff) * np.log2(1 + np.abs(diff))
            diff = diff_transformed
            cbar_label = cbar_label.replace("Difference", "Difference (signed log2)")
        
        save_path = output_path / f"{prefix}_difference_{comp_name}.png"
        
        try:
            if brain_projection is None:
                logger.warning(f"brain_projection function not available, skipping {comp_name}")
                file_paths[comp_name] = None
            else:
                # Use diverging colormap for differences
                brain_projection(
                    config,
                    diff,
                    str(save_path),
                    cbar_label=cbar_label,
                    cbar_endlabels=(f"{group2} > {group1}", f"{group1} > {group2}")
                )
                file_paths[comp_name] = str(save_path)
                logger.info(f"Saved difference map {comp_name} to {save_path}")
        except Exception as e:
            logger.error(f"Error creating difference map {comp_name}: {e}")
            file_paths[comp_name] = None
    
    return file_paths


def plot_significance_maps(
    significance_map: np.ndarray,
    config: Dict,
    output_path: str,
    title: str = "Significant Parcels (FDR-corrected)",
    prefix: str = "sdi"
) -> str:
    """
    Create brain projection showing significant parcels.
    
    Parameters:
    -----------
    significance_map : np.ndarray
        Binary significance map (1 = significant, 0 = not)
    config : dict
        Configuration dictionary
    output_path : str
        Output file path
    title : str
        Plot title
    prefix : str
        File prefix
    
    Returns:
    --------
    file_path : str
        Path to saved figure
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if brain_projection is None:
            logger.warning("brain_projection function not available, skipping significance map")
            return None
        brain_projection(
            config,
            significance_map.astype(float),
            str(output_file),
            cbar_label="Significance",
            cbar_endlabels=("Not Significant", "Significant")
        )
        logger.info(f"Saved significance map to {output_file}")
        return str(output_file)
    except Exception as e:
        logger.error(f"Error creating significance map: {e}")
        return None


def plot_parcel_heatmap(
    parcel_stats: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    config: Optional[Dict] = None,
    significance_map: Optional[np.ndarray] = None,
    use_log_scale: bool = True,
    use_median: bool = False,
    vmin_percentile: Optional[float] = 1.0,
    vmax_percentile: Optional[float] = 99.0,
    clip_outliers: bool = True,
    show_outlier_info: bool = True
) -> plt.Figure:
    """
    Create heatmap of SDI values (parcels × groups) with spatial organization.
    
    Parameters:
    -----------
    parcel_stats : pd.DataFrame
        Parcel-level statistics with mean/median SDI per group
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    config : dict, optional
        Configuration dictionary for parcel ordering
    significance_map : np.ndarray, optional
        Binary significance map (1 = significant, 0 = not) for overlay
    use_log_scale : bool
        Whether to use log2 transformation (default: True)
    use_median : bool
        Whether to use median instead of mean (default: False, uses mean)
    vmin_percentile : float, optional
        Percentile to use for vmin (default: 1.0). If None, uses full range minimum.
    vmax_percentile : float, optional
        Percentile to use for vmax (default: 99.0). If None, uses full range maximum.
    clip_outliers : bool
        Whether to clip outlier values to the percentile range (default: True).
        If False, only the color scale is clipped but outliers remain visible.
    show_outlier_info : bool
        Whether to display information about clipped outliers (default: True)
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    # Extract group columns (mean or median)
    if use_median:
        group_cols = [col for col in parcel_stats.columns if col.startswith('median_')]
        stat_name = 'median'
    else:
        group_cols = [col for col in parcel_stats.columns if col.startswith('mean_')]
        stat_name = 'mean'
    
    # Fallback to mean if median columns don't exist
    if len(group_cols) == 0:
        group_cols = [col for col in parcel_stats.columns if col.startswith('mean_')]
        stat_name = 'mean'
        use_median = False
    
    groups = [col.replace(f'{stat_name}_', '') for col in group_cols]
    
    # Create matrix
    heatmap_data = parcel_stats[group_cols].values
    
    # Apply log transformation if requested
    if use_log_scale:
        heatmap_data = np.log2(heatmap_data + 1e-10)
        cbar_label = f'SDI Value (log2)'
    else:
        cbar_label = f'SDI Value ({stat_name})'
    
    # Calculate color scale limits using percentiles to handle outliers
    valid_data = heatmap_data[np.isfinite(heatmap_data)]
    if len(valid_data) > 0:
        if vmin_percentile is not None:
            vmin = np.percentile(valid_data, vmin_percentile)
        else:
            vmin = np.nanmin(heatmap_data)
        
        if vmax_percentile is not None:
            vmax = np.percentile(valid_data, vmax_percentile)
        else:
            vmax = np.nanmax(heatmap_data)
        
        # Store actual min/max for reporting
        actual_min = np.nanmin(heatmap_data)
        actual_max = np.nanmax(heatmap_data)
        
        # Clip outliers if requested
        n_outliers = 0
        if clip_outliers:
            # Count outliers before clipping
            n_outliers = np.sum((heatmap_data < vmin) | (heatmap_data > vmax))
            if n_outliers > 0:
                logger.info(f"Clipping {n_outliers} outlier values ({100*n_outliers/heatmap_data.size:.1f}%) "
                           f"to percentile range [{vmin_percentile:.1f}, {vmax_percentile:.1f}]")
            # Clip values to percentile range
            heatmap_data = np.clip(heatmap_data, vmin, vmax)
        elif actual_min < vmin or actual_max > vmax:
            # Count outliers even if not clipping (for info display)
            n_outliers = np.sum((heatmap_data < vmin) | (heatmap_data > vmax))
            logger.info(f"Color scale clipped to [{vmin:.2f}, {vmax:.2f}] "
                       f"but {n_outliers} outlier values ({100*n_outliers/heatmap_data.size:.1f}%) remain visible")
    else:
        vmin, vmax = 0, 1
        actual_min, actual_max = 0, 1
        n_outliers = 0
    
    # Order parcels by region if config provided
    if config is not None:
        try:
            from .spatial_analysis import get_parcel_ordering_by_region
            ordering = get_parcel_ordering_by_region(config=config)
            # Reorder heatmap data
            parcel_ids = parcel_stats['parcel_id'].values
            reordered_indices = np.argsort(ordering)[np.searchsorted(ordering, parcel_ids)]
            heatmap_data = heatmap_data[reordered_indices]
            parcel_ids_ordered = parcel_ids[reordered_indices]
        except Exception as e:
            logger.warning(f"Could not apply spatial ordering: {e}")
            parcel_ids_ordered = parcel_stats['parcel_id'].values
    else:
        parcel_ids_ordered = parcel_stats['parcel_id'].values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with explicit vmin/vmax to handle outliers
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    
    # Add significance overlay if provided
    if significance_map is not None:
        # Ensure significance map matches parcel order
        if len(significance_map) == len(heatmap_data):
            # Create overlay for significant parcels
            for i, is_sig in enumerate(significance_map):
                if is_sig:
                    # Add asterisk or highlight
                    for j in range(len(groups)):
                        ax.text(j, i, '*', ha='center', va='center', 
                               color='white', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Parcel ID (ordered by region)', fontsize=12)
    ax.set_title(f'SDI Values Across Parcels and Groups ({stat_name.capitalize()})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    
    # Set y-axis ticks (show every 50th parcel)
    y_ticks = np.arange(0, len(parcel_ids_ordered), 50)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(parcel_ids_ordered[y_ticks])
    
    # Colorbar with value range
    cbar = plt.colorbar(im, ax=ax)
    
    # Create informative colorbar label
    if vmin_percentile is not None or vmax_percentile is not None:
        # Show both displayed range and actual range
        range_text = f'displayed: [{vmin:.2f}, {vmax:.2f}]'
        if actual_min < vmin or actual_max > vmax:
            range_text += f'\nactual: [{actual_min:.2f}, {actual_max:.2f}]'
        cbar.set_label(f'{cbar_label}\n({range_text})', 
                       fontsize=10, rotation=270, labelpad=25)
    else:
        cbar.set_label(f'{cbar_label}\n(range: {vmin:.2f} to {vmax:.2f})', 
                       fontsize=11, rotation=270, labelpad=25)
    
    # Add notes about significance and outliers
    note_texts = []
    if significance_map is not None:
        note_texts.append('* = Significant (FDR-corrected)')
    
    if note_texts:
        note_text = '\n'.join(note_texts)
        ax.text(0.02, 0.98, note_text, 
               transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved parcel heatmap to {output_path}")
    
    return fig


def plot_effect_sizes(
    posthoc_results: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    always_generate: bool = True
) -> plt.Figure:
    """
    Create forest plot of effect sizes with confidence intervals and interpretation guide.
    
    Parameters:
    -----------
    posthoc_results : pd.DataFrame
        Post-hoc test results with effect sizes and CIs
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    always_generate : bool
        Always generate plot even if no significant results (default: True)
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object (None if no data and always_generate=False)
    """
    if len(posthoc_results) == 0:
        if not always_generate:
            logger.warning("No posthoc results provided for effect size plot")
            return None
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No post-hoc test results available', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Effect Sizes for Group Comparisons', fontsize=14, fontweight='bold')
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    comparisons = [f"{row['group1']} vs {row['group2']}" 
                   for _, row in posthoc_results.iterrows()]
    effect_sizes = posthoc_results['effect_size'].values
    ci_lower = posthoc_results['ci_lower'].values
    ci_upper = posthoc_results['ci_upper'].values
    
    # Get effect size name
    if 'effect_size_name' in posthoc_results.columns:
        effect_size_name = posthoc_results['effect_size_name'].iloc[0]
    else:
        effect_size_name = "Effect Size"
    
    y_pos = np.arange(len(comparisons))
    
    # Plot effect sizes with error bars
    ax.errorbar(effect_sizes, y_pos, xerr=[effect_sizes - ci_lower, ci_upper - effect_sizes],
               fmt='o', capsize=5, capthick=2, markersize=8, linewidth=2)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='No effect')
    
    # Add effect size interpretation guide (Cohen's conventions for Cohen's d)
    if 'Cohen' in effect_size_name or 'd' in effect_size_name.lower():
        # Cohen's d conventions
        ax.axvline(x=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=-0.2, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=-0.8, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        
        # Add interpretation text
        ax.text(0.98, 0.02, 
               'Effect size interpretation:\n'
               '|d| < 0.2: Negligible\n'
               '0.2 ≤ |d| < 0.5: Small\n'
               '0.5 ≤ |d| < 0.8: Medium\n'
               '|d| ≥ 0.8: Large',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparisons)
    ax.set_xlabel(f'{effect_size_name} (95% CI)', fontsize=12)
    ax.set_title('Effect Sizes for Group Comparisons', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved effect size plot to {output_path}")
    
    return fig


def plot_yeo7_network_comparison(
    network_stats: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    use_median: bool = True,
    use_log_scale: bool = True,
    show_error_bars: bool = True,
    plot_type: str = 'bar'  # 'bar' or 'violin'
) -> plt.Figure:
    """
    Create plot comparing SDI across Yeo7 networks and groups.
    
    Parameters:
    -----------
    network_stats : pd.DataFrame
        Network-level statistics (must have 'yeo7_network', 'yeo7_network_name', 'group', 'mean', 'median', 'std')
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    use_median : bool
        Use median instead of mean (default: True)
    use_log_scale : bool
        Use log2 scale for y-axis (default: True)
    show_error_bars : bool
        Show error bars (default: True)
    plot_type : str
        'bar' or 'violin' (default: 'bar')
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    if len(network_stats) == 0:
        logger.warning("No network statistics provided")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get networks and groups
    networks = sorted(network_stats['yeo7_network'].unique())
    network_names = []
    for net in networks:
        net_data = network_stats[network_stats['yeo7_network'] == net]
        if len(net_data) > 0 and 'yeo7_network_name' in net_data.columns:
            name = net_data['yeo7_network_name'].iloc[0]
            network_names.append(name if pd.notna(name) else f'Network {net}')
        else:
            network_names.append(f'Network {net}')
    
    groups = sorted(network_stats['group'].unique())
    
    # Determine statistic
    stat_col = 'median' if use_median else 'mean'
    if stat_col not in network_stats.columns:
        stat_col = 'mean'
        use_median = False
    
    if plot_type == 'bar':
        x = np.arange(len(networks))
        width = 0.25
        
        for i, group in enumerate(groups):
            group_data = network_stats[network_stats['group'] == group]
            
            values = []
            errors = []
            
            for net in networks:
                net_data = group_data[group_data['yeo7_network'] == net]
                if len(net_data) > 0:
                    val = net_data[stat_col].values[0]
                    values.append(val)
                    
                    if show_error_bars and 'std' in net_data.columns:
                        errors.append(net_data['std'].values[0])
                    else:
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # Apply log transformation
            if use_log_scale:
                values = [np.log2(v + 1e-10) if v > 0 else np.nan for v in values]
            
            color = GROUP_COLORS.get(group, '#95A5A6')
            
            if show_error_bars and any(e > 0 for e in errors):
                ax.bar(x + i * width, values, width, label=group, color=color, alpha=0.7,
                      yerr=errors, capsize=3, error_kw={'linewidth': 1.5})
            else:
                ax.bar(x + i * width, values, width, label=group, color=color, alpha=0.7)
        
        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(network_names, rotation=45, ha='right')
    else:
        # Violin plot (would need individual parcel data, not implemented here)
        logger.warning("Violin plot for networks requires individual parcel data. Using bar plot instead.")
        return plot_yeo7_network_comparison(network_stats, output_path, figsize, 
                                           use_median, use_log_scale, show_error_bars, 'bar')
    
    ylabel = f'SDI Value ({stat_col.capitalize()})'
    if use_log_scale:
        ylabel += ' (log2)'
    
    ax.set_xlabel('Yeo7 Network', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'SDI Values by Yeo7 Network and Group ({stat_col.capitalize()})', 
                fontsize=14, fontweight='bold')
    
    if use_log_scale:
        ax.set_yscale('log')
    
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Yeo7 network comparison plot to {output_path}")
    
    return fig


def plot_regional_comparison(
    region_stats: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    use_median: bool = True,
    use_log_scale: bool = True,
    show_error_bars: bool = True,
    statistical_results: Optional[pd.DataFrame] = None
) -> plt.Figure:
    """
    Create bar plot comparing SDI across regions and groups.
    
    Parameters:
    -----------
    region_stats : pd.DataFrame
        Regional statistics (must have 'mean', 'median', 'std', 'q25', 'q75' columns)
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    use_median : bool
        Use median instead of mean (default: True, recommended)
    use_log_scale : bool
        Use log2 scale for y-axis (default: True)
    show_error_bars : bool
        Show error bars (IQR or 95% CI) (default: True)
    statistical_results : pd.DataFrame, optional
        Statistical test results for annotations
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    regions = sorted(region_stats['region_category'].unique())
    groups = sorted(region_stats['group'].unique())
    
    x = np.arange(len(regions))
    width = 0.25
    
    # Determine which statistic to use
    stat_col = 'median' if use_median else 'mean'
    if stat_col not in region_stats.columns:
        stat_col = 'mean'  # Fallback
        use_median = False
    
    # Determine error bars
    if show_error_bars:
        if 'q25' in region_stats.columns and 'q75' in region_stats.columns:
            error_type = 'iqr'
        elif 'std' in region_stats.columns:
            error_type = 'std'
        else:
            error_type = None
    else:
        error_type = None
    
    for i, group in enumerate(groups):
        group_data = region_stats[region_stats['group'] == group]
        
        # Get central tendency values and quartiles (store originals before transformation)
        values_original = []
        q25_values = []
        q75_values = []
        std_values = []
        
        for r in regions:
            region_data = group_data[group_data['region_category'] == r]
            if len(region_data) > 0:
                val = region_data[stat_col].values[0]
                values_original.append(val)
                
                # Store quartiles and std for proper transformation
                if error_type == 'iqr':
                    q25 = region_data['q25'].values[0] if 'q25' in region_data.columns else val
                    q75 = region_data['q75'].values[0] if 'q75' in region_data.columns else val
                    q25_values.append(q25)
                    q75_values.append(q75)
                    std_values.append(None)  # Not used for IQR
                elif error_type == 'std':
                    std_val = region_data['std'].values[0] if 'std' in region_data.columns else 0
                    q25_values.append(None)
                    q75_values.append(None)
                    std_values.append(std_val)
                else:
                    q25_values.append(None)
                    q75_values.append(None)
                    std_values.append(None)
            else:
                values_original.append(0)
                q25_values.append(None)
                q75_values.append(None)
                std_values.append(None)
        
        # Apply log transformation if needed
        if use_log_scale:
            # Transform values to log space
            values = [np.log2(v + 1e-10) if v > 0 else np.nan for v in values_original]
            
            # Transform error bars properly: transform q25/q75 to log space first, then calculate differences
            errors_lower = []
            errors_upper = []
            
            for val_orig, val_log, q25, q75, std_val in zip(values_original, values, q25_values, q75_values, std_values):
                if error_type == 'iqr' and q25 is not None and q75 is not None:
                    # Transform quartiles to log space, then calculate error bars
                    q25_log = np.log2(q25 + 1e-10) if q25 > 0 else np.nan
                    q75_log = np.log2(q75 + 1e-10) if q75 > 0 else np.nan
                    
                    if not np.isnan(val_log) and not np.isnan(q25_log):
                        errors_lower.append(val_log - q25_log)  # Distance from median to q25 in log space
                    else:
                        errors_lower.append(0)
                    
                    if not np.isnan(val_log) and not np.isnan(q75_log):
                        errors_upper.append(q75_log - val_log)  # Distance from q75 to median in log space
                    else:
                        errors_upper.append(0)
                elif error_type == 'std' and std_val is not None and std_val > 0 and val_orig > 0:
                    # For std with log scale: approximate transformation
                    # log2(x + std) - log2(x) ≈ std / (x * ln(2)) for small std/x
                    # Use symmetric error bars
                    error_size = std_val / (val_orig * np.log(2))
                    errors_lower.append(error_size)
                    errors_upper.append(error_size)
                else:
                    errors_lower.append(0)
                    errors_upper.append(0)
        else:
            # No log transformation: calculate error bars in original space
            values = values_original
            errors_lower = []
            errors_upper = []
            for val, q25, q75, std_val in zip(values, q25_values, q75_values, std_values):
                if error_type == 'iqr' and q25 is not None and q75 is not None:
                    errors_lower.append(val - q25)
                    errors_upper.append(q75 - val)
                elif error_type == 'std' and std_val is not None:
                    errors_lower.append(std_val)
                    errors_upper.append(std_val)
                else:
                    errors_lower.append(0)
                    errors_upper.append(0)
        
        color = GROUP_COLORS.get(group, '#95A5A6')
        
        # Plot bars with error bars
        if error_type:
            ax.bar(x + i * width, values, width, label=group, color=color, alpha=0.7,
                  yerr=[errors_lower, errors_upper], capsize=3, error_kw={'linewidth': 1.5})
        else:
            ax.bar(x + i * width, values, width, label=group, color=color, alpha=0.7)
    
    # Add statistical annotations if provided
    if statistical_results is not None and len(statistical_results) > 0:
        y_max = max([v for vals in [values] for v in vals if not np.isnan(v)])
        y_range = y_max - min([v for vals in [values] for v in vals if not np.isnan(v)])
        y_pos = y_max + 0.1 * y_range
        
        # Add significance markers (simplified - would need region-specific results)
        for idx, row in statistical_results.iterrows():
            # This is a placeholder - would need region-specific p-values
            pass
    
    # Set labels and title
    ylabel = f'SDI Value ({stat_col.capitalize()})'
    if use_log_scale:
        ylabel += ' (log2)'
    
    ax.set_xlabel('Region Category', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'SDI Values by Region and Group ({stat_col.capitalize()})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    
    if use_log_scale:
        ax.set_yscale('log')
    
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved regional comparison plot to {output_path}")
    
    return fig


def plot_outlier_analysis(
    unified_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    outlier_method: str = 'iqr'
) -> plt.Figure:
    """
    Create visualization of outlier distribution across groups and parcels.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    outlier_method : str
        'iqr' or 'zscore' (default: 'iqr')
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure object
    """
    from .descriptive_stats import detect_outliers_iqr
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    groups = sorted(unified_df['group'].unique())
    
    # Panel 1: Outlier count per group
    ax1 = axes[0, 0]
    outlier_counts = []
    for group in groups:
        outliers = detect_outliers_iqr(unified_df, group=group)
        outlier_counts.append(len(outliers))
    
    colors = [GROUP_COLORS.get(g, '#95A5A6') for g in groups]
    bars = ax1.bar(groups, outlier_counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Outliers', fontsize=11)
    ax1.set_title('Outlier Count per Group', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, outlier_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Outlier distribution by parcel (heatmap)
    ax2 = axes[0, 1]
    outlier_matrix = np.zeros((len(groups), 430))
    
    for i, group in enumerate(groups):
        outliers = detect_outliers_iqr(unified_df, group=group)
        if len(outliers) > 0:
            for _, row in outliers.iterrows():
                parcel_id = int(row['parcel_id'])
                outlier_matrix[i, parcel_id] = 1
    
    im = ax2.imshow(outlier_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
    ax2.set_xlabel('Parcel ID', fontsize=11)
    ax2.set_ylabel('Group', fontsize=11)
    ax2.set_yticks(range(len(groups)))
    ax2.set_yticklabels(groups)
    ax2.set_title('Outlier Spatial Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Outlier (1=yes, 0=no)')
    
    # Panel 3: Outlier values distribution
    ax3 = axes[1, 0]
    for group in groups:
        outliers = detect_outliers_iqr(unified_df, group=group)
        if len(outliers) > 0:
            outlier_values = outliers['sdi_value'].values
            ax3.hist(np.log2(outlier_values + 1e-10), bins=20, alpha=0.6, 
                    label=group, color=GROUP_COLORS.get(group, '#95A5A6'))
    
    ax3.set_xlabel('SDI Value (log2)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Outlier Values', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Outlier Analysis Summary\n" + "="*50 + "\n\n"
    for group in groups:
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        outliers = detect_outliers_iqr(unified_df, group=group)
        
        summary_text += f"{group}:\n"
        summary_text += f"  Total parcels: {len(group_data)}\n"
        summary_text += f"  Outliers: {len(outliers)} ({100*len(outliers)/len(group_data):.1f}%)\n"
        if len(outliers) > 0:
            summary_text += f"  Outlier range: {outliers['sdi_value'].min():.2e} to {outliers['sdi_value'].max():.2e}\n"
        summary_text += "\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=9,
            family='monospace', verticalalignment='top')
    
    plt.suptitle('Outlier Analysis Across Groups', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved outlier analysis plot to {output_path}")
    
    return fig

