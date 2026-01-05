# SDI Analysis Report: bandpass

**Generated:** 2025-12-22 16:56:00

## Data Overview

**Groups analyzed:** Healthy Controls, ICU, ICU Delirium

| Group | N Subjects | N Parcels | N Observations |
|-------|------------|-----------|----------------|
| Healthy Controls | 12 | 430 | 5160 |
| ICU | 5 | 430 | 2150 |
| ICU Delirium | 12 | 430 | 5160 |

## Statistical Assumption Checks

**Recommended test type:** Kruskal-Wallis
**All groups normal:** False
**Variances homogeneous:** False

### Normality Tests

| group | n | test_used | statistic | pvalue | is_normal |
| --- | --- | --- | --- | --- | --- |
| ICU | 2150 | shapiro_wilk | 0.6449236869812012 | 0.0 | False |
| ICU Delirium | 5160 | dagostino_pearson | 4472.381960554028 | 0.0 | False |
| Healthy Controls | 5160 | dagostino_pearson | 4780.237870709923 | 0.0 | False |

### Variance Tests

**Levene test:**
- Statistic: 131.55219940367553
- p-value: 2.907272224790196e-57
- Homogeneous: False

**Bartlett test:**
- Statistic: 2453.5766951031565
- p-value: 0.0
- Homogeneous: False

## Group-Level Comparisons

### Primary Test

**Test:** Kruskal-Wallis
**Statistic:** 36.57261848378403
**p-value:** 1.1438176713446912e-08

### Post-hoc Tests

| group1 | group2 | pvalue_corrected | effect_size | significant |
| --- | --- | --- | --- | --- |
| Healthy Controls | ICU | 0.0000 | 0.0696 | True |
| Healthy Controls | ICU Delirium | 0.0000 | 0.0603 | True |
| ICU | ICU Delirium | 0.3914 | -0.0127 | False |

## Parcel-Level Comparisons

## Spatial Analysis

### Regional Statistics

| region_category | mean | std | n_parcels |
| --- | --- | --- | --- |
| Amygdala | 1.3909409292605652 | 0.24345239227729024 | 2 |
| Basal Ganglia | 1.3794760950012648 | 0.5156437891979816 | 18 |
| Brainstem | 1.3524108746866077 | 0.355978338641563 | 4 |
| Cerebellum | 1.289233670249786 | 0.23813422042665255 | 2 |
| Default Mode | 1.3848999920250857 | 1.0569242902586868 | 80 |
| Dorsal Attention | 1.022206179828233 | 0.37629853828747944 | 43 |
| Frontoparietal | 1.0171592270532612 | 0.3510158443920921 | 45 |
| Limbic | 1.26859053198026 | 0.5978964282526434 | 26 |
| Other | 0.9340200323407651 | 0.1242155711848843 | 2 |
| Somatomotor | 1.1296661536249504 | 0.6242492460605943 | 54 |
| Thalamus | 2.3186482477986927 | 1.9026766627545992 | 42 |
| Ventral Attention | 1.1557889521679598 | 0.6263345839488245 | 46 |
| Visual | 0.9405819598704802 | 0.46556737043631746 | 66 |

**Significant regions (p < 0.05):** 10

| region_category | test | pvalue |
| --- | --- | --- |
| Amygdala | Kruskal-Wallis | 0.0003776653599486843 |
| Basal Ganglia | Kruskal-Wallis | 1.546387369209907e-32 |
| Cerebellum | Kruskal-Wallis | 1.0192428006299092e-07 |
| Default Mode | Kruskal-Wallis | 0.0005456977228259522 |
| Frontoparietal | Kruskal-Wallis | 0.008574793682598779 |
| Limbic | Kruskal-Wallis | 0.03681828921040536 |
| Other | Kruskal-Wallis | 0.006999660511525263 |
| Somatomotor | Kruskal-Wallis | 0.009291110806446512 |
| Thalamus | Kruskal-Wallis | 1.8659086994400906e-50 |
| Visual | Kruskal-Wallis | 5.772232276274228e-10 |

## Generated Figures

### Assumption Checks

- [sdi_bandpass_qq_plots.png](figures/assumption_checks/sdi_bandpass_qq_plots.png)
- [sdi_bandpass_variance_comparison.png](figures/assumption_checks/sdi_bandpass_variance_comparison.png)

### Distributions

- [sdi_bandpass_distributions.png](figures/distributions/sdi_bandpass_distributions.png)

### Comparisons

- [sdi_bandpass_effect_sizes.png](figures/comparisons/sdi_bandpass_effect_sizes.png)
- [sdi_bandpass_group_comparison.png](figures/comparisons/sdi_bandpass_group_comparison.png)
- [sdi_bandpass_parcel_heatmap.png](figures/comparisons/sdi_bandpass_parcel_heatmap.png)
- [sdi_bandpass_regional_comparison.png](figures/comparisons/sdi_bandpass_regional_comparison.png)

### Brain Projections

- [sdi_bandpass_brain_projection_Healthy_Controls.png](figures/brain_projections/sdi_bandpass_brain_projection_Healthy_Controls.png)
- [sdi_bandpass_brain_projection_ICU.png](figures/brain_projections/sdi_bandpass_brain_projection_ICU.png)
- [sdi_bandpass_brain_projection_ICU_Delirium.png](figures/brain_projections/sdi_bandpass_brain_projection_ICU_Delirium.png)
- [sdi_bandpass_difference_ICU_Delirium_vs_Healthy_Controls.png](figures/brain_projections/sdi_bandpass_difference_ICU_Delirium_vs_Healthy_Controls.png)
- [sdi_bandpass_difference_ICU_vs_Healthy_Controls.png](figures/brain_projections/sdi_bandpass_difference_ICU_vs_Healthy_Controls.png)
- [sdi_bandpass_difference_ICU_vs_ICU_Delirium.png](figures/brain_projections/sdi_bandpass_difference_ICU_vs_ICU_Delirium.png)

## Output Files

### Descriptive Statistics

- [sdi_bandpass_distribution_stats.csv](descriptive_stats/sdi_bandpass_distribution_stats.csv)
- [sdi_bandpass_group_level_stats.csv](descriptive_stats/sdi_bandpass_group_level_stats.csv)
- [sdi_bandpass_outliers.csv](descriptive_stats/sdi_bandpass_outliers.csv)
- [sdi_bandpass_parcel_level_stats.csv](descriptive_stats/sdi_bandpass_parcel_level_stats.csv)

### Statistical Tests

- [sdi_bandpass_normality_tests.csv](statistical_tests/sdi_bandpass_normality_tests.csv)
- [sdi_bandpass_parcel_comparisons.csv](statistical_tests/sdi_bandpass_parcel_comparisons.csv)
- [sdi_bandpass_posthoc_tests.csv](statistical_tests/sdi_bandpass_posthoc_tests.csv)
- [sdi_bandpass_primary_test.csv](statistical_tests/sdi_bandpass_primary_test.csv)
- [sdi_bandpass_regional_stats.csv](statistical_tests/sdi_bandpass_regional_stats.csv)
- [sdi_bandpass_regional_tests.csv](statistical_tests/sdi_bandpass_regional_tests.csv)
- [sdi_bandpass_significant_parcels_bonferroni.csv](statistical_tests/sdi_bandpass_significant_parcels_bonferroni.csv)
- [sdi_bandpass_significant_parcels_bonferroni_only.csv](statistical_tests/sdi_bandpass_significant_parcels_bonferroni_only.csv)
- [sdi_bandpass_significant_parcels_fdr.csv](statistical_tests/sdi_bandpass_significant_parcels_fdr.csv)
- [sdi_bandpass_significant_parcels_fdr_only.csv](statistical_tests/sdi_bandpass_significant_parcels_fdr_only.csv)
- [sdi_bandpass_test_recommendations.csv](statistical_tests/sdi_bandpass_test_recommendations.csv)
- [sdi_bandpass_variance_tests.csv](statistical_tests/sdi_bandpass_variance_tests.csv)

### Reports

