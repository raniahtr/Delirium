"""
Utilities package for Delirium connectome analysis.

This package provides common utilities for:
- Configuration and paths
- Connectome loading and processing
- Subject iteration
- Plotting functions
- QC data handling
"""

from .connectome_utils import (
    # Configuration
    ROOT_DIR,
    PREPROC_DIR,
    ATLAS_DIR,
    FINAL_ATLAS_PATH,
    PARCELS_LABELS_XLSX,
    YEO7_PATH,
    YEO7_RESAMP_PATH,
    ATLAS_TO_YEO7_CSV,
    N_ROIS,
    CONNECTOME_TYPES,
    DEFAULT_GROUPS,
    
    # Subject utilities
    find_subjects,
    find_subjects_with_connectomes,
    iterate_subjects,
    
    # SNR
    load_snr_qc_data,
    
    # Connectome loading
    load_connectome,
    load_all_connectomes,
    compute_group_average,
    
    # Cache management
    get_cache_info,
    clear_connectome_cache,
    enable_connectome_cache,
    preload_connectomes,
    
    # Pattern finding
    find_connectome_files,
    get_connectome_path,
   
    
    # Plotting
    setup_plotting_style,
    plot_log_connectome,
    plot_all_connectomes,
    plot_group_comparison_boxplot,
    plot_connectome_heatmap,
    get_group_cmap,
    get_group_colors,
    
    # Statistical utilities
    detect_outliers,
    
    # Helpers
    normalize_connectome,
    prepare_matrix,
    
    # Structural connectome similarity analysis
    get_region_category_indices,
    extract_region_submatrix,
    compute_sc_similarity_between_subjects,
    load_parcellation_mappings,
    compute_overall_average_connectome,
    plot_overall_average_connectome,
    compute_full_subject_similarity_matrix,
    plot_subject_similarity_heatmap,
    recover_sc_similarity_matrix,
    plot_sc_fingerprint_btw_subjects_by_group,
    compute_sc_similarity_all_subjects_by_region,
    plot_sc_similarity_all_subjects_by_region,
    
    # Connectivity strength analysis
    compute_connectivity_strength_metrics,
    run_connectivity_statistical_tests,
    plot_connectivity_strength_results,
    run_full_connectivity_analysis,
    
    # Efficiency/path length analysis
    compute_efficiency_pathlength,
    compute_efficiency_pathlength_metrics,
    run_efficiency_statistical_tests,
    plot_efficiency_pathlength_results,
    run_full_efficiency_analysis,
    
    # Bootstrap analysis
    bootstrap_global_network_metrics,
    plot_bootstrap_results,
    save_bootstrap_summary,
    
    # Connectome ordering and visualization
    build_cortical_order_LH_RH,
    build_subcortical_order_by_category,
    build_combined_order_cortical_subcortical,
    plot_group_connectome_LH_RH,
    plot_subcortical_connectome,
    plot_combined_connectome_cortical_subcortical,
    plot_multi_group_connectomes,
    YEO7_COLORS,
    SUBCORTICAL_CATEGORY_COLORS,
)

__all__ = [
    # Configuration
    'ROOT_DIR',
    'PREPROC_DIR',
    'ATLAS_DIR',
    'FINAL_ATLAS_PATH',
    'PARCELS_LABELS_XLSX',
    'YEO7_PATH',
    'YEO7_RESAMP_PATH',
    'ATLAS_TO_YEO7_CSV',
    'N_ROIS',
    'CONNECTOME_TYPES',
    'DEFAULT_GROUPS',
    
    # Functions
    'find_subjects',
    'find_subjects_with_connectomes',
    'iterate_subjects',
    'load_connectome',
    'load_all_connectomes',
    'compute_group_average',
    
    # Cache management
    'get_cache_info',
    'clear_connectome_cache',
    'enable_connectome_cache',
    'preload_connectomes',
    
    'find_connectome_files',
    'get_connectome_path',
    'setup_plotting_style',
    'plot_log_connectome',
    'plot_all_connectomes',
    'plot_group_comparison_boxplot',
    'plot_connectome_heatmap',
    'get_group_colors',
    'detect_outliers',
    'normalize_connectome',
    'prepare_matrix',
    
    # Structural connectome similarity analysis
    'get_region_category_indices',
    'extract_region_submatrix',
    'compute_sc_similarity_between_subjects',
    'load_parcellation_mappings',
    'compute_overall_average_connectome',
    'plot_overall_average_connectome',
    'compute_full_subject_similarity_matrix',
    'plot_subject_similarity_heatmap',
    'recover_sc_similarity_matrix',
    'plot_sc_fingerprint_btw_subjects_by_group',
    'compute_sc_similarity_all_subjects_by_region',
    'plot_sc_similarity_all_subjects_by_region',
    
    # Connectivity strength analysis
    'compute_connectivity_strength_metrics',
    'run_connectivity_statistical_tests',
    'plot_connectivity_strength_results',
    'run_full_connectivity_analysis',
    
    # Efficiency/path length analysis
    'compute_efficiency_pathlength',
    'compute_efficiency_pathlength_metrics',
    'run_efficiency_statistical_tests',
    'plot_efficiency_pathlength_results',
    'run_full_efficiency_analysis',
    
    # Bootstrap analysis
    'bootstrap_global_network_metrics',
    'plot_bootstrap_results',
    'save_bootstrap_summary',
    
    # Connectome ordering and visualization
    'build_cortical_order_LH_RH',
    'build_subcortical_order_by_category',
    'build_combined_order_cortical_subcortical',
    'plot_group_connectome_LH_RH',
    'plot_subcortical_connectome',
    'plot_combined_connectome_cortical_subcortical',
    'plot_multi_group_connectomes',
    'YEO7_COLORS',
    'SUBCORTICAL_CATEGORY_COLORS',
]

