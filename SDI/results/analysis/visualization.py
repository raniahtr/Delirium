"""
Visualization functions for SDI analysis.

All figures use the standard color scheme and MLP parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

# Import MLP params and color scheme
from config import mlp_param
from utils.group import get_group_color

# Apply MLP params
plt.rcParams.update(mlp_param.MPL_PARAMS)

logger = logging.getLogger(__name__)

# Color mapping: standard names -> colors
COLOR_MAPPING = {
    'ICU': get_group_color()['CR'],
    'ICU Delirium': get_group_color()['Delirium'],
    'Healthy Controls': get_group_color()['HC']
}


def plot_distributions(
    unified_df: pd.DataFrame,
    output_path: str,
    combine_plots: bool = True,
    use_log_scale: bool = True,
    percentile_clip: Tuple[float, float] = (1, 99),
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create distribution plots (violin + box plots) per group.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    output_path : str
        Path to save figure
    combine_plots : bool
        Combine violin and box plots (default: True)
    use_log_scale : bool
        Use log2 transformation (default: True)
    percentile_clip : tuple
        Percentiles for clipping outliers (default: (1, 99))
    figsize : tuple
        Figure size (default: (12, 8))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = sorted(unified_df['group'].unique())
    colors = [COLOR_MAPPING.get(g, '#808080') for g in groups]
    
    # Prepare data
    plot_data = []
    for group in groups:
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        
        if use_log_scale:
            group_data_clean = np.log2(group_data_clean + 1e-10)
        
        # Clip outliers
        if percentile_clip:
            lower = np.percentile(group_data_clean, percentile_clip[0])
            upper = np.percentile(group_data_clean, percentile_clip[1])
            group_data_clean = group_data_clean[(group_data_clean >= lower) & (group_data_clean <= upper)]
        
        for val in group_data_clean:
            plot_data.append({'group': group, 'sdi_value': val})
    
    plot_df = pd.DataFrame(plot_data)
    
    if combine_plots:
        # Violin plot with box plot overlay
        parts = ax.violinplot(
            [plot_df[plot_df['group'] == g]['sdi_value'].values for g in groups],
            positions=range(len(groups)),
            showmeans=False,
            showmedians=True,
            widths=0.7
        )
        
        # Color violins
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Box plot overlay
        bp = ax.boxplot(
            [plot_df[plot_df['group'] == g]['sdi_value'].values for g in groups],
            positions=range(len(groups)),
            widths=0.3,
            patch_artist=True,
            showfliers=False
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
    else:
        # Just box plot
        bp = ax.boxplot(
            [plot_df[plot_df['group'] == g]['sdi_value'].values for g in groups],
            labels=groups,
            patch_artist=True
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_xticklabels(groups)
    ylabel = "SDI (log2)" if use_log_scale else "SDI Value"
    ax.set_ylabel(ylabel)
    ax.set_title("SDI Distribution by Group")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def plot_group_comparison(
    unified_df: pd.DataFrame,
    posthoc_results: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    use_log_scale: bool = True,
    percentile_clip: Tuple[float, float] = (1, 99)
):
    """
    Create group comparison box plot with statistical annotations.
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified SDI DataFrame
    posthoc_results : pd.DataFrame, optional
        Post-hoc test results
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    use_log_scale : bool
        Use log2 transformation
    percentile_clip : tuple
        Percentiles for clipping
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = sorted(unified_df['group'].unique())
    colors = [COLOR_MAPPING.get(g, '#808080') for g in groups]
    
    # Prepare data
    plot_data = []
    for group in groups:
        group_data = unified_df[unified_df['group'] == group]['sdi_value'].values
        group_data_clean = group_data[np.isfinite(group_data)]
        
        if use_log_scale:
            group_data_clean = np.log2(group_data_clean + 1e-10)
        
        if percentile_clip:
            lower = np.percentile(group_data_clean, percentile_clip[0])
            upper = np.percentile(group_data_clean, percentile_clip[1])
            group_data_clean = group_data_clean[(group_data_clean >= lower) & (group_data_clean <= upper)]
        
        for val in group_data_clean:
            plot_data.append({'group': group, 'sdi_value': val})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Box plot
    bp = ax.boxplot(
        [plot_df[plot_df['group'] == g]['sdi_value'].values for g in groups],
        labels=groups,
        patch_artist=True,
        widths=0.6
    )
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add statistical annotations if available
    if posthoc_results is not None and len(posthoc_results) > 0:
        y_max = max([np.max(plot_df[plot_df['group'] == g]['sdi_value'].values) for g in groups])
        y_range = max([np.max(plot_df[plot_df['group'] == g]['sdi_value'].values) - 
                       np.min(plot_df[plot_df['group'] == g]['sdi_value'].values) for g in groups])
        
        y_pos = y_max + 0.1 * y_range
        line_height = 0.05 * y_range
        
        for _, row in posthoc_results.iterrows():
            if row.get('significant', False):
                group1_idx = groups.index(row['group1'])
                group2_idx = groups.index(row['group2'])
                
                # Draw bracket
                x1, x2 = group1_idx + 1, group2_idx + 1
                ax.plot([x1, x1, x2, x2], [y_pos, y_pos + line_height, y_pos + line_height, y_pos], 'k-', lw=1.5)
                
                # Add p-value
                pval = row.get('pvalue_corrected', row.get('pvalue', 1.0))
                if pval < 0.001:
                    sig_text = '***'
                elif pval < 0.01:
                    sig_text = '**'
                elif pval < 0.05:
                    sig_text = '*'
                else:
                    sig_text = f'p={pval:.3f}'
                
                ax.text((x1 + x2) / 2, y_pos + line_height + 0.01 * y_range, sig_text,
                       ha='center', va='bottom', fontsize=12)
                
                y_pos += 0.15 * y_range
    
    ylabel = "SDI (log2)" if use_log_scale else "SDI Value"
    ax.set_ylabel(ylabel)
    ax.set_title("Group Comparison")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved group comparison plot to {output_path}")
    plt.close()


def plot_effect_sizes(
    posthoc_results: pd.DataFrame,
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create effect size plot (forest plot style).
    
    Parameters:
    -----------
    posthoc_results : pd.DataFrame
        Post-hoc test results with effect sizes
    output_path : str
        Path to save figure
    figsize : tuple
        Figure size
    """
    if len(posthoc_results) == 0:
        logger.warning("No post-hoc results to plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    comparisons = []
    effect_sizes = []
    ci_lowers = []
    ci_uppers = []
    colors_list = []
    
    for _, row in posthoc_results.iterrows():
        comp = f"{row['group1']} vs {row['group2']}"
        comparisons.append(comp)
        
        # Use effect size CIs if available
        if 'effect_size_ci_lower' in row and 'effect_size_ci_upper' in row and pd.notna(row.get('effect_size_ci_lower')):
            effect_sizes.append(row.get('effect_size', 0))
            ci_lowers.append(row.get('effect_size_ci_lower', 0))
            ci_uppers.append(row.get('effect_size_ci_upper', 0))
        else:
            # No CI available (e.g., Cliff's delta) - just plot effect size without error bars
            effect_sizes.append(row.get('effect_size', 0))
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
        
        # Color based on significance
        if row.get('significant', False):
            colors_list.append('#C0392B')  # Red for significant
        else:
            colors_list.append('#7F8C8D')  # Gray for non-significant
    
    y_pos = np.arange(len(comparisons))
    
    # Plot error bars only where CIs are available
    has_ci = ~(np.isnan(ci_lowers) | np.isnan(ci_uppers))
    
    if np.any(has_ci):
        # Plot comparisons with CIs (error bars)
        effect_sizes_with_ci = np.array(effect_sizes)[has_ci]
        y_pos_with_ci = y_pos[has_ci]
        ci_lowers_valid = np.array(ci_lowers)[has_ci]
        ci_uppers_valid = np.array(ci_uppers)[has_ci]
        colors_with_ci = np.array(colors_list)[has_ci]
        
        # Plot error bars (this also plots the points)
        ax.errorbar(effect_sizes_with_ci, y_pos_with_ci, 
                   xerr=[effect_sizes_with_ci - ci_lowers_valid,
                         ci_uppers_valid - effect_sizes_with_ci],
                   fmt='none', capsize=5, capthick=2, color='black', zorder=2, alpha=0.7)
        
        # Color points with CIs
        for es, y, color in zip(effect_sizes_with_ci, y_pos_with_ci, colors_with_ci):
            ax.plot(es, y, 'o', color=color, markersize=10, zorder=3)
    
    # Plot comparisons without CIs (just points, no error bars)
    if np.any(~has_ci):
        effect_sizes_no_ci = np.array(effect_sizes)[~has_ci]
        y_pos_no_ci = y_pos[~has_ci]
        colors_no_ci = np.array(colors_list)[~has_ci]
        
        for es, y, color in zip(effect_sizes_no_ci, y_pos_no_ci, colors_no_ci):
            ax.plot(es, y, 'o', color=color, markersize=10, zorder=3, alpha=0.7)
    
    # Vertical line at 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparisons)
    ax.set_xlabel("Effect Size")
    ax.set_title("Effect Sizes with 95% Confidence Intervals")
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved effect sizes plot to {output_path}")


def plot_regional_comparison(
    region_stats_df: pd.DataFrame,
    output_path: str,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Create regional comparison bar plot.
    
    Parameters:
    -----------
    region_stats_df : pd.DataFrame
        DataFrame with regional statistics
    output_path : str
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = sorted(region_stats_df['group'].unique())
    regions = sorted(region_stats_df['region_category'].unique())
    
    x = np.arange(len(regions))
    width = 0.25
    
    colors = [COLOR_MAPPING.get(g, '#808080') for g in groups]
    
    for i, group in enumerate(groups):
        group_data = region_stats_df[region_stats_df['group'] == group]
        means = []
        sems = []
        
        for region in regions:
            region_data = group_data[group_data['region_category'] == region]
            if len(region_data) > 0:
                means.append(region_data['mean'].iloc[0])
                sems.append(region_data['sem'].iloc[0])
            else:
                means.append(0)
                sems.append(0)
        
        ax.bar(x + i * width, means, width, label=group, color=colors[i], alpha=0.7, yerr=sems, capsize=5)
    
    ax.set_xlabel('Region Category')
    ax.set_ylabel('Mean SDI')
    ax.set_title('Regional Comparison by Group')
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved regional comparison plot to {output_path}")


def plot_parcel_heatmap(
    parcel_results_df: pd.DataFrame,
    output_path: str,
    use_corrected: bool = True,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Create heatmap of parcel-level p-values.
    
    This shows the omnibus test results (Kruskal-Wallis or ANOVA) across all groups
    for each parcel. It represents which parcels show significant group differences
    overall, not group-specific comparisons.
    
    Parameters:
    -----------
    parcel_results_df : pd.DataFrame
        DataFrame with parcel-level results (omnibus test across all groups)
    output_path : str
        Path to save figure
    use_corrected : bool
        Use corrected p-values if available
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    p_col = 'pvalue_corrected' if (use_corrected and 'pvalue_corrected' in parcel_results_df.columns) else 'pvalue'
    
    # Create -log10(p) values
    p_values = parcel_results_df[p_col].values
    p_values_clean = np.where(np.isfinite(p_values) & (p_values > 0), -np.log10(p_values), 0)
    
    # Reshape to 430 parcels
    n_parcels = 430
    heatmap_data = np.zeros(n_parcels)
    
    for _, row in parcel_results_df.iterrows():
        parcel_id = int(row['parcel_id'])
        if 1 <= parcel_id <= n_parcels:
            idx = parcel_results_df.index[parcel_results_df['parcel_id'] == parcel_id].tolist()
            if len(idx) > 0:
                heatmap_data[parcel_id - 1] = p_values_clean[idx[0]]
    
    # Create heatmap (1D as bar plot or 2D if we want spatial arrangement)
    im = ax.imshow(heatmap_data.reshape(1, -1), aspect='auto', cmap='hot_r', interpolation='nearest')
    ax.set_xlabel('Parcel ID')
    ax.set_ylabel('')
    ax.set_title(f'Parcel-level Significance Map (-log10(p), {"corrected" if use_corrected else "uncorrected"})')
    ax.set_yticks([])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('-log10(p-value)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved parcel heatmap to {output_path}")


def plot_brain_projections(
    group_arrays: Dict[str, np.ndarray],
    config: Dict,
    output_dir: str,
    prefix: str = "sdi",
    use_log2: bool = True,
    use_median: bool = False
) -> Dict[str, str]:
    """
    Create brain projection maps for each group.
    
    Parameters:
    -----------
    group_arrays : dict
        Dictionary mapping group names to SDI arrays (n_subjects, 430) or (430,)
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory
    prefix : str
        File prefix
    use_log2 : bool
        Use log2 transformation
    use_median : bool
        Use median instead of mean for individual data
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping group names to file paths
    """
    from plot.brain_projection import brain_projection
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Process arrays: if 2D (individual data), compute group average
    processed_arrays = {}
    for group_name, sdi_array in group_arrays.items():
        if sdi_array.ndim == 2:
            # Individual data: compute group average
            if use_median:
                processed_arrays[group_name] = np.median(sdi_array, axis=0)
            else:
                processed_arrays[group_name] = np.mean(sdi_array, axis=0)
        else:
            # Already 1D (group-averaged)
            processed_arrays[group_name] = sdi_array
    
    for group_name, sdi_array in processed_arrays.items():
        # Transform values
        if use_log2:
            transformed_sdi = np.log2(sdi_array + 1e-10)
            cbar_label = "SDI (log2)"
        else:
            transformed_sdi = sdi_array
            cbar_label = "SDI Value"
        
        save_path = output_path / f"{prefix}_brain_projection_{group_name.replace(' ', '_')}.png"
        
        try:
            brain_projection(
                config,
                transformed_sdi,
                str(save_path),
                cbar_label=cbar_label,
                max_abs=1,  # Use fixed -1 to 1 scale for consistent visualization
                cbar_endlabels=("Coupling", "Decoupling")
            )
            file_paths[group_name] = str(save_path)
            logger.info(f"Saved brain projection for {group_name} to {save_path}")
        except Exception as e:
            logger.error(f"Error creating brain projection for {group_name}: {e}")
            file_paths[group_name] = None
    
    return file_paths


def plot_difference_maps(
    group_arrays: Dict[str, np.ndarray],
    config: Dict,
    output_dir: str,
    prefix: str = "sdi",
    use_log2: bool = True,
    use_median: bool = False
) -> Dict[str, str]:
    """
    Create difference maps between groups.
    
    Parameters:
    -----------
    group_arrays : dict
        Dictionary mapping group names to SDI arrays
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory
    prefix : str
        File prefix
    use_log2 : bool
        Use log2 transformation
    use_median : bool
        Use median instead of mean
    
    Returns:
    --------
    file_paths : dict
        Dictionary mapping comparison names to file paths
    """
    from plot.brain_projection import brain_projection
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Process arrays
    processed_arrays = {}
    for group_name, sdi_array in group_arrays.items():
        if sdi_array.ndim == 2:
            processed_arrays[group_name] = np.median(sdi_array, axis=0) if use_median else np.mean(sdi_array, axis=0)
        else:
            processed_arrays[group_name] = sdi_array
    
    groups = list(processed_arrays.keys())
    
    # Create all pairwise comparisons
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            arr1 = processed_arrays[group1]
            arr2 = processed_arrays[group2]
            
            # Handle NaN/inf values
            arr1_clean = np.where(np.isfinite(arr1), arr1, 0)
            arr2_clean = np.where(np.isfinite(arr2), arr2, 0)
            
            # Calculate difference
            diff = arr1_clean - arr2_clean
            
            if use_log2:
                # Use signed log transform
                diff_transformed = np.sign(diff) * np.log2(1 + np.abs(diff))
                diff = diff_transformed
                cbar_label = f"SDI Difference (log2): {group1} - {group2}"
            else:
                cbar_label = f"SDI Difference: {group1} - {group2}"
            
            comp_name = f"{group1.replace(' ', '_')}_vs_{group2.replace(' ', '_')}"
            save_path = output_path / f"{prefix}_difference_{comp_name}.png"
            
            # Create clear labels: higher SDI = higher decoupling
            # diff = group1 - group2
            # Negative values (left): group2 has higher SDI (more decoupling)
            # Positive values (right): group1 has higher SDI (more decoupling)
            lower_label = f"Higher SDI in {group2}"
            higher_label = f"Higher SDI in {group1}"
            
            try:
                brain_projection(
                    config,
                    diff,
                    str(save_path),
                    cbar_label=cbar_label,
                    max_abs=1,  # Use fixed -1 to 1 scale
                    cbar_endlabels=(lower_label, higher_label)
                )
                file_paths[comp_name] = str(save_path)
                logger.info(f"Saved difference map {comp_name} to {save_path}")
            except Exception as e:
                logger.error(f"Error creating difference map {comp_name}: {e}")
                file_paths[comp_name] = None
    
    return file_paths

