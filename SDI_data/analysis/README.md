# SDI Analysis Pipeline

A comprehensive, modular analysis pipeline for Structural Decoupling Index (SDI) data following Van de Ville and Preti methodology.

## Overview

This pipeline provides:
- **Descriptive statistics** (group-level and parcel-level)
- **Statistical assumption verification** (normality, homogeneity of variance)
- **Group-level comparisons** (Kruskal-Wallis, ANOVA, post-hoc tests)
- **Parcel-level comparisons** with multiple comparisons correction (FDR, Bonferroni)
- **Spatial analysis** (regional and network-level aggregation)
- **Publication-ready visualizations** using the standard color scheme

## Color Scheme

All figures use the following color scheme:
- **ICU**: #E67E22 (orange)
- **ICU Delirium**: #C0392B (dark red/orange)
- **Healthy Controls**: #3498DB (blue)

## Quick Start

### Run Full Pipeline

```python
from utils.load_config import load_config
from analysis.run_full_analysis import run_full_analysis

config = load_config()
results = run_full_analysis(
    config=config,
    band='allFB',
    use_parametric=False,  # Recommended: use non-parametric tests
    alpha=0.05,
    correction_method='fdr_bh',
    generate_plots=True
)
```

### Command Line

```bash
cd github/SDI_delirium
python -m analysis.run_full_analysis --band allFB --alpha 0.05
```

## Module Structure

### 1. `load_sdi_data.py`
Loads SDI h5 files for all groups and creates unified DataFrames.

**Key functions:**
- `load_all_sdi_data(config, band='allFB')`: Load all SDI data
- `validate_sdi_data(df, group_arrays)`: Validate data integrity

### 2. `descriptive_stats.py`
Computes descriptive statistics and distribution analysis.

**Key functions:**
- `compute_group_level_stats(df)`: Group-level statistics
- `compute_parcel_level_stats(df)`: Parcel-level statistics
- `detect_outliers_iqr(df)`: Outlier detection
- `export_descriptive_stats(...)`: Export to CSV

### 3. `assumption_checks.py`
Verifies statistical assumptions and recommends appropriate tests.

**Key functions:**
- `check_all_assumptions(df)`: Comprehensive assumption checks
- `test_normality(data)`: Normality tests (Shapiro-Wilk, D'Agostino-Pearson)
- `test_homogeneity_of_variance(group_data)`: Variance tests (Levene, Bartlett)
- `plot_qq_plots(df)`: Q-Q plots for normality assessment

### 4. `group_comparisons.py`
Performs group-level statistical comparisons.

**Key functions:**
- `perform_group_comparisons(df, use_parametric=False)`: Main comparison function
- `perform_kruskal_wallis_test(df)`: Non-parametric test
- `perform_anova_test(df)`: Parametric test
- `perform_posthoc_tests(df, test_type='mannwhitneyu')`: Post-hoc pairwise tests
- `bootstrap_confidence_interval(...)`: Bootstrap CIs for effect sizes

### 5. `parcel_comparisons.py`
Performs parcel-level comparisons with multiple comparisons correction.

**Key functions:**
- `perform_parcel_level_comparisons(df, ...)`: Compare all 430 parcels
- `create_significance_map(parcel_results, correction_method='fdr_bh')`: Binary significance map
- `create_continuous_significance_map(...)`: Continuous -log10(p) map

### 6. `spatial_analysis.py`
Regional and network-level spatial analysis.

**Key functions:**
- `aggregate_by_region(df)`: Aggregate by region categories
- `aggregate_by_yeo7_network(df)`: Aggregate by Yeo7 networks
- `analyze_spatial_clustering(significance_map)`: Spatial clustering analysis

### 7. `visualization.py`
All plotting functions using the standard color scheme.

**Key functions:**
- `plot_distributions(df)`: Violin plots and boxplots
- `plot_group_comparison(df, posthoc_results)`: Boxplots with statistical annotations
- `plot_brain_projections(group_arrays, config)`: Brain projection maps per group
- `plot_difference_maps(group_arrays, config)`: Group difference maps
- `plot_significance_maps(significance_map, config)`: Significance maps
- `plot_effect_sizes(posthoc_results)`: Forest plots of effect sizes

### 8. `run_full_analysis.py`
Main pipeline orchestrator.

**Key function:**
- `run_full_analysis(config, ...)`: Run complete pipeline
- `generate_analysis_report(results, ...)`: Generate markdown report

## Output Structure

```
results/analysis/
├── descriptive_stats/
│   ├── sdi_allFB_group_level_stats.csv
│   ├── sdi_allFB_parcel_level_stats.csv
│   └── sdi_allFB_distribution_stats.csv
├── statistical_tests/
│   ├── sdi_allFB_primary_test.csv
│   ├── sdi_allFB_posthoc_tests.csv
│   ├── sdi_allFB_parcel_comparisons.csv
│   ├── sdi_allFB_significant_parcels_fdr.csv
│   └── sdi_allFB_normality_tests.csv
├── figures/
│   ├── distributions/
│   ├── comparisons/
│   ├── brain_projections/
│   ├── significance_maps/
│   └── assumption_checks/
└── reports/
    └── sdi_allFB_analysis_report.md
```

## Examples

See `example_run_analysis.py` for detailed examples:
1. Running the full pipeline
2. Running individual modules
3. Custom analysis with specific parameters
4. Visualization-only workflow

## Statistical Approach

The pipeline follows a **non-parametric first** approach:

1. **Assumption checks**: Verify normality and homogeneity of variance
2. **Primary test**: 
   - If assumptions met: One-way ANOVA
   - If assumptions not met: Kruskal-Wallis (default)
3. **Post-hoc tests**:
   - Parametric: Welch's t-test with FDR correction
   - Non-parametric: Mann-Whitney U with FDR correction
4. **Effect sizes**:
   - Parametric: Cohen's d
   - Non-parametric: Cliff's delta
5. **Multiple comparisons**: FDR (Benjamini-Hochberg) and Bonferroni

## Dependencies

- `pandas`, `numpy`, `scipy`
- `matplotlib`, `seaborn`
- `statsmodels` (for multiple comparisons correction)
- `nilearn` (for brain projections, via `plot.brain_projection`)





