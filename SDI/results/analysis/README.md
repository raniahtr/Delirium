This analysis was originally in another notebook, putting it here for reference 
Github reference https://github.com/Samik29sm/SDI_delirium.git
# SDI Analysis Pipeline

Comprehensive statistical analysis pipeline for Structural Decoupling Index (SDI) data.

## Quick Start

### From Terminal

The analysis can be run directly from the terminal:

```bash
# Basic usage (default: bandpass, non-parametric tests, FDR correction)
python -m analysis.run_analysis

# Specify frequency band
python -m analysis.run_analysis --band bandpass

# Use parametric tests (if assumptions are met)
python -m analysis.run_analysis --band bandpass --use-parametric

# Custom significance level and correction method
python -m analysis.run_analysis --band bandpass --alpha 0.01 --correction bonferroni

# Skip plot generation (faster for testing)
python -m analysis.run_analysis --band bandpass --no-plots

# Custom output directory
python -m analysis.run_analysis --band bandpass --output-dir /path/to/output
```

### Using the Wrapper Script

```bash
# From the github/SDI_delirium directory
./run_sdi_analysis.sh --band bandpass

# Or with full path
/path/to/github/SDI_delirium/run_sdi_analysis.sh --band bandpass
```

### From Python

```python
from utils.load_config import load_config
from analysis.run_analysis import run_analysis

config = load_config()
results = run_analysis(
    config=config,
    band='bandpass',
    use_parametric=False,
    alpha=0.05,
    correction_method='fdr_bh',
    generate_plots=True
)
```

## Command-Line Options

- `--band BAND`: Frequency band to analyze (default: `bandpass`)
- `--output-dir OUTPUT_DIR`: Output directory (default: `results/analysis`)
- `--use-parametric`: Use parametric tests (ANOVA, t-test) instead of non-parametric
- `--alpha ALPHA`: Significance level (default: `0.05`)
- `--correction {fdr_bh,bonferroni}`: Multiple comparisons correction method (default: `fdr_bh`)
- `--no-plots`: Skip generating plots
- `--pipeline-type {HC_ICU}`: Pipeline type (default: `HC_ICU`)

## Output Structure

```
results/analysis/
├── descriptive_stats/
│   ├── sdi_bandpass_group_level_stats.csv
│   ├── sdi_bandpass_parcel_level_stats.csv
│   ├── sdi_bandpass_distribution_stats.csv
│   └── sdi_bandpass_outliers.csv
├── statistical_tests/
│   ├── sdi_bandpass_normality_tests.csv
│   ├── sdi_bandpass_variance_tests.csv
│   ├── sdi_bandpass_test_recommendations.csv
│   ├── sdi_bandpass_primary_test.csv
│   ├── sdi_bandpass_posthoc_tests.csv
│   ├── sdi_bandpass_parcel_comparisons.csv
│   ├── sdi_bandpass_significant_parcels_fdr.csv
│   ├── sdi_bandpass_significant_parcels_bonferroni.csv
│   └── sdi_bandpass_regional_stats.csv
├── figures/
│   ├── assumption_checks/
│   │   ├── sdi_bandpass_qq_plots.png
│   │   └── sdi_bandpass_variance_comparison.png
│   ├── distributions/
│   │   └── sdi_bandpass_distributions.png
│   ├── comparisons/
│   │   ├── sdi_bandpass_group_comparison.png
│   │   ├── sdi_bandpass_effect_sizes.png
│   │   ├── sdi_bandpass_regional_comparison.png
│   │   └── sdi_bandpass_parcel_heatmap.png
│   └── brain_projections/
│       ├── sdi_bandpass_brain_projection_ICU.png
│       ├── sdi_bandpass_brain_projection_ICU_Delirium.png
│       ├── sdi_bandpass_brain_projection_Healthy_Controls.png
│       └── sdi_bandpass_difference_*.png
└── reports/
    └── sdi_bandpass_analysis_report.md
```

## Statistical Methodology

1. **Assumption checks**: Normality (Shapiro-Wilk/D'Agostino-Pearson) and homogeneity of variance (Levene/Bartlett)
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

## Color Scheme

All figures use the standard color scheme:
- **ICU**: #E67E22 (orange)
- **ICU Delirium**: #C0392B (dark red/orange)
- **Healthy Controls**: #3498DB (blue)

## Requirements

- Python 3.7+
- pandas, numpy, scipy
- matplotlib, seaborn
- statsmodels
- nilearn (for brain projections)
- openpyxl (for Excel file reading)




