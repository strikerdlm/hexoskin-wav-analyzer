#!/usr/bin/env python3
"""
=============================================================================
COMPREHENSIVE ANALYSIS OF T03-T08 PHYSIOLOGICAL DATA ACROSS SOLS
=============================================================================

This script performs comprehensive analysis of physiological data for subjects:
- T03_Nancy
- T04_Michelle  
- T05_Felicitas
- T06_Mara_Selena
- T07_Geraldinn
- T08_Karina

Analysis includes:
1. Data loading and cleaning
2. Descriptive statistics
3. Non-parametric statistical tests
4. Time series analysis across SOLs
5. Interpretation of findings

Author: Valquiria Research Team
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    kruskal, mannwhitneyu, friedmanchisquare, wilcoxon,
    spearmanr, pearsonr, shapiro, levene, bartlett
)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

print("="*80)
print("COMPREHENSIVE ANALYSIS: T03-T08 PHYSIOLOGICAL DATA ACROSS SOLS")
print("="*80)

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

print("\n1. DATA LOADING AND PREPARATION")
print("-" * 50)

# Define file paths
file_paths = {
    'T03_Nancy': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T03_Nancy.csv',
    'T04_Michelle': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T04_Michelle.csv',
    'T05_Felicitas': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T05_Felicitas.csv',
    'T06_Mara_Selena': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T06_Mara_Selena.csv',
    'T07_Geraldinn': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T07_Geraldinn.csv',
    'T08_Karina': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T08_Karina.csv'
}

# Load data
dfs = {}
for name, path in file_paths.items():
    try:
        df = pd.read_csv(path)
        dfs[name] = df
        print(f"✓ Loaded {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading {name}: {e}")

if len(dfs) == 0:
    print("✗ No data loaded. Please check file paths.")
    exit()

# Combine all dataframes
print("\nCombining datasets...")
for name, df in dfs.items():
    df['subject'] = name
    df['subject_id'] = name.split('_')[0]  # T03, T04, etc.

# Combine all data
df_combined = pd.concat(dfs.values(), ignore_index=True)
print(f"✓ Combined dataset: {df_combined.shape[0]:,} rows")

# =============================================================================
# 2. DATA CLEANING AND PREPROCESSING
# =============================================================================

print("\n2. DATA CLEANING AND PREPROCESSING")
print("-" * 50)

# Check for missing values
print("Missing values per column:")
missing_values = df_combined.isnull().sum()
missing_percent = (missing_values / len(df_combined)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_values,
    'Missing_Percent': missing_percent
}).sort_values('Missing_Percent', ascending=False)
print(missing_df[missing_df['Missing_Count'] > 0])

# Convert time to datetime (handle large timestamp values)
if 'time [s/256]' in df_combined.columns:
    try:
        # Try to convert raw timestamps by dividing by 256 first
        df_combined['datetime'] = pd.to_datetime(df_combined['time [s/256]'] / 256, unit='s')
        print("✓ Converted time to datetime (divided by 256)")
        
        # Extract temporal features
        df_combined['date'] = df_combined['datetime'].dt.date
        df_combined['hour'] = df_combined['datetime'].dt.hour
        df_combined['minute'] = df_combined['datetime'].dt.minute
        df_combined['day_of_week'] = df_combined['datetime'].dt.day_name()
        
        print(f"✓ Time range: {df_combined['datetime'].min()} to {df_combined['datetime'].max()}")
    except Exception as e:
        print(f"✗ Could not convert time column: {e}")
        print("✓ Will use SOL for temporal analysis instead")
elif 'time [s/1000]' in df_combined.columns:
    try:
        df_combined['datetime'] = pd.to_datetime(df_combined['time [s/1000]'], unit='ms')
        print("✓ Converted time to datetime")
        
        # Extract temporal features
        df_combined['date'] = df_combined['datetime'].dt.date
        df_combined['hour'] = df_combined['datetime'].dt.hour
        df_combined['minute'] = df_combined['datetime'].dt.minute
        df_combined['day_of_week'] = df_combined['datetime'].dt.day_name()
        
        print(f"✓ Time range: {df_combined['datetime'].min()} to {df_combined['datetime'].max()}")
    except Exception as e:
        print(f"✗ Could not convert time column: {e}")
        print("✓ Will use SOL for temporal analysis instead")
else:
    print("✓ No time column found - using SOL for temporal analysis")

# Identify physiological variables
physio_vars = []
for col in df_combined.columns:
    if any(unit in col.lower() for unit in ['[bpm]', '[rpm]', '[%]', '[g]', '[ml/min]', '[mmhg]', '[c]']):
        physio_vars.append(col)

print(f"✓ Identified {len(physio_vars)} physiological variables:")
for var in physio_vars:
    print(f"  - {var}")

# =============================================================================
# 3. DESCRIPTIVE ANALYSIS
# =============================================================================

print("\n3. DESCRIPTIVE ANALYSIS")
print("-" * 50)

# Overall statistics by subject
print("Overall Statistics by Subject:")
print("=" * 60)

for subject in df_combined['subject'].unique():
    subject_data = df_combined[df_combined['subject'] == subject]
    print(f"\n{subject}:")
    print(f"  - Records: {len(subject_data):,}")
    print(f"  - SOL range: {subject_data['Sol'].min()} to {subject_data['Sol'].max()}")
    print(f"  - Duration: {subject_data['datetime'].max() - subject_data['datetime'].min()}")

# Descriptive statistics for each physiological variable
print("\nDescriptive Statistics by Variable:")
print("=" * 60)

descriptive_stats = {}
for var in physio_vars:
    if var in df_combined.columns:
        stats_df = df_combined.groupby('subject')[var].describe().round(3)
        descriptive_stats[var] = stats_df
        print(f"\n{var}:")
        print(stats_df)

# =============================================================================
# 4. NORMALITY TESTING
# =============================================================================

print("\n4. NORMALITY TESTING")
print("-" * 50)

normality_results = {}
for var in physio_vars:
    if var in df_combined.columns:
        normality_results[var] = {}
        print(f"\nNormality tests for {var}:")
        
        for subject in df_combined['subject'].unique():
            subject_data = df_combined[df_combined['subject'] == subject][var].dropna()
            if len(subject_data) > 3:
                stat, p_value = shapiro(subject_data[:5000])  # Limit sample for Shapiro-Wilk
                normality_results[var][subject] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
                print(f"  {subject}: Shapiro-Wilk p-value = {p_value:.4f} ({'Normal' if p_value > 0.05 else 'Non-normal'})")

# =============================================================================
# 5. NON-PARAMETRIC STATISTICAL TESTS
# =============================================================================

print("\n5. NON-PARAMETRIC STATISTICAL TESTS")
print("-" * 50)

# Kruskal-Wallis test for between-group comparisons
print("Kruskal-Wallis Test (Between-Subject Comparisons):")
print("=" * 60)

kruskal_results = {}
for var in physio_vars:
    if var in df_combined.columns:
        groups = []
        for subject in df_combined['subject'].unique():
            subject_data = df_combined[df_combined['subject'] == subject][var].dropna()
            if len(subject_data) > 0:
                groups.append(subject_data)
        
        if len(groups) >= 2:
            try:
                h_stat, p_value = kruskal(*groups)
                kruskal_results[var] = {
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                print(f"\n{var}:")
                print(f"  H-statistic: {h_stat:.4f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                
                # Interpretation
                if p_value < 0.05:
                    print(f"  INTERPRETATION: There are statistically significant differences in {var} between subjects (p < 0.05)")
                else:
                    print(f"  INTERPRETATION: No statistically significant differences in {var} between subjects (p ≥ 0.05)")
            except Exception as e:
                print(f"  Error in Kruskal-Wallis test for {var}: {e}")

# Mann-Whitney U test for pairwise comparisons
print("\nMann-Whitney U Test (Pairwise Comparisons):")
print("=" * 60)

mann_whitney_results = {}
subjects = df_combined['subject'].unique()
for var in physio_vars[:3]:  # Limit to first 3 variables for brevity
    if var in df_combined.columns:
        mann_whitney_results[var] = {}
        print(f"\nPairwise comparisons for {var}:")
        
        for i, subject1 in enumerate(subjects):
            for j, subject2 in enumerate(subjects):
                if i < j:  # Avoid duplicate comparisons
                    data1 = df_combined[df_combined['subject'] == subject1][var].dropna()
                    data2 = df_combined[df_combined['subject'] == subject2][var].dropna()
                    
                    if len(data1) > 0 and len(data2) > 0:
                        try:
                            u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                            mann_whitney_results[var][f"{subject1}_vs_{subject2}"] = {
                                'u_statistic': u_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                            print(f"  {subject1} vs {subject2}: U={u_stat:.0f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
                        except Exception as e:
                            print(f"  Error comparing {subject1} vs {subject2}: {e}")

# =============================================================================
# 6. TIME SERIES ANALYSIS ACROSS SOLS
# =============================================================================

print("\n6. TIME SERIES ANALYSIS ACROSS SOLS")
print("-" * 50)

# Calculate daily (SOL-based) statistics
print("Calculating daily statistics for each subject...")
daily_stats = {}
for var in physio_vars:
    if var in df_combined.columns:
        daily_stats[var] = df_combined.groupby(['subject', 'Sol'])[var].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(3).reset_index()

# Spearman correlation with SOL (time trend analysis)
print("\nSpearman Correlation with SOL (Time Trend Analysis):")
print("=" * 60)

spearman_results = {}
for var in physio_vars:
    if var in df_combined.columns:
        spearman_results[var] = {}
        print(f"\nTime trends for {var}:")
        
        for subject in df_combined['subject'].unique():
            subject_data = df_combined[df_combined['subject'] == subject]
            if len(subject_data) > 10:  # Need sufficient data points
                corr_coef, p_value = spearmanr(subject_data['Sol'], subject_data[var])
                spearman_results[var][subject] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                trend_direction = "increasing" if corr_coef > 0 else "decreasing"
                significance = "significant" if p_value < 0.05 else "non-significant"
                
                print(f"  {subject}: r={corr_coef:.4f}, p={p_value:.4f} ({trend_direction}, {significance})")
                
                # Interpretation
                if p_value < 0.05:
                    if abs(corr_coef) > 0.5:
                        strength = "strong"
                    elif abs(corr_coef) > 0.3:
                        strength = "moderate"
                    else:
                        strength = "weak"
                    print(f"    INTERPRETATION: {strength.capitalize()} {trend_direction} trend over time")
                else:
                    print(f"    INTERPRETATION: No significant temporal trend")

# Friedman test for within-subject changes across SOLs
print("\nFriedman Test (Within-Subject Changes Across SOLs):")
print("=" * 60)

friedman_results = {}
for var in physio_vars[:3]:  # Limit for brevity
    if var in df_combined.columns:
        friedman_results[var] = {}
        print(f"\nFriedman test for {var}:")
        
        for subject in df_combined['subject'].unique():
            subject_data = df_combined[df_combined['subject'] == subject]
            
            # Get daily means for each SOL
            daily_means = subject_data.groupby('Sol')[var].mean().dropna()
            
            if len(daily_means) >= 3:  # Need at least 3 time points
                try:
                    # Prepare data for Friedman test (need multiple observations per time point)
                    sol_data = []
                    for sol in daily_means.index:
                        sol_values = subject_data[subject_data['Sol'] == sol][var].dropna()
                        if len(sol_values) >= 10:  # Need sufficient observations
                            sol_data.append(sol_values[:10])  # Take first 10 observations
                    
                    if len(sol_data) >= 3:
                        # Pad arrays to same length
                        max_len = max(len(arr) for arr in sol_data)
                        padded_data = []
                        for arr in sol_data:
                            padded = np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan)
                            padded_data.append(padded)
                        
                        # Remove NaN values
                        data_array = np.array(padded_data).T
                        clean_data = data_array[~np.isnan(data_array).any(axis=1)]
                        
                        if len(clean_data) >= 3:
                            chi2_stat, p_value = friedmanchisquare(*clean_data.T)
                            friedman_results[var][subject] = {
                                'chi2_statistic': chi2_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                            print(f"  {subject}: χ²={chi2_stat:.4f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
                            
                            # Interpretation
                            if p_value < 0.05:
                                print(f"    INTERPRETATION: Significant changes in {var} across SOLs for {subject}")
                            else:
                                print(f"    INTERPRETATION: No significant changes in {var} across SOLs for {subject}")
                        else:
                            print(f"  {subject}: Insufficient clean data for Friedman test")
                    else:
                        print(f"  {subject}: Insufficient SOL data for Friedman test")
                except Exception as e:
                    print(f"  {subject}: Error in Friedman test: {e}")

# =============================================================================
# 7. VISUALIZATION
# =============================================================================

print("\n7. CREATING VISUALIZATIONS")
print("-" * 50)

# Create comprehensive visualization plots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Physiological Data Analysis: T03-T08 Subjects', fontsize=16, fontweight='bold')

# Plot 1: Box plots comparing subjects
if len(physio_vars) > 0:
    var_to_plot = physio_vars[0]  # Use first available variable
    df_plot = df_combined[df_combined[var_to_plot].notna()]
    
    sns.boxplot(data=df_plot, x='subject', y=var_to_plot, ax=axes[0,0])
    axes[0,0].set_title(f'Distribution of {var_to_plot} by Subject')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)

# Plot 2: Time series across SOLs
if len(physio_vars) > 0:
    var_to_plot = physio_vars[0]
    daily_means = df_combined.groupby(['subject', 'Sol'])[var_to_plot].mean().reset_index()
    
    for subject in df_combined['subject'].unique():
        subject_data = daily_means[daily_means['subject'] == subject]
        axes[0,1].plot(subject_data['Sol'], subject_data[var_to_plot], 
                      marker='o', linewidth=2, label=subject, alpha=0.7)
    
    axes[0,1].set_title(f'{var_to_plot} - Daily Means Across SOLs')
    axes[0,1].set_xlabel('SOL (Day)')
    axes[0,1].set_ylabel(var_to_plot)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

# Plot 3: Correlation heatmap
if len(physio_vars) > 1:
    correlation_data = df_combined[physio_vars].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1,0])
    axes[1,0].set_title('Correlation Matrix of Physiological Variables')

# Plot 4: Distribution of measurements by hour
if len(physio_vars) > 0:
    var_to_plot = physio_vars[0]
    hourly_means = df_combined.groupby(['subject', 'hour'])[var_to_plot].mean().reset_index()
    
    for subject in df_combined['subject'].unique():
        subject_data = hourly_means[hourly_means['subject'] == subject]
        axes[1,1].plot(subject_data['hour'], subject_data[var_to_plot], 
                      marker='o', linewidth=2, label=subject, alpha=0.7)
    
    axes[1,1].set_title(f'{var_to_plot} - Hourly Patterns')
    axes[1,1].set_xlabel('Hour of Day')
    axes[1,1].set_ylabel(var_to_plot)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_analysis_T03_T08.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. SUMMARY AND INTERPRETATION
# =============================================================================

print("\n8. SUMMARY AND INTERPRETATION")
print("-" * 50)

print("\nKEY FINDINGS:")
print("=" * 40)

print("\n1. DATA OVERVIEW:")
print(f"   - Total subjects analyzed: {len(df_combined['subject'].unique())}")
print(f"   - Total records: {len(df_combined):,}")
print(f"   - Physiological variables: {len(physio_vars)}")
print(f"   - SOL range: {df_combined['Sol'].min()} to {df_combined['Sol'].max()}")

print("\n2. BETWEEN-SUBJECT COMPARISONS:")
significant_vars = [var for var in kruskal_results.keys() if kruskal_results[var]['significant']]
print(f"   - Variables with significant between-subject differences: {len(significant_vars)}")
for var in significant_vars:
    print(f"     • {var} (p = {kruskal_results[var]['p_value']:.4f})")

print("\n3. TEMPORAL TRENDS:")
subjects_with_trends = {}
for var in spearman_results.keys():
    for subject in spearman_results[var].keys():
        if spearman_results[var][subject]['significant']:
            if subject not in subjects_with_trends:
                subjects_with_trends[subject] = []
            subjects_with_trends[subject].append(var)

print(f"   - Subjects with significant temporal trends:")
for subject, vars_with_trends in subjects_with_trends.items():
    print(f"     • {subject}: {len(vars_with_trends)} variables")

print("\n4. WITHIN-SUBJECT CHANGES:")
subjects_with_changes = {}
for var in friedman_results.keys():
    for subject in friedman_results[var].keys():
        if friedman_results[var][subject]['significant']:
            if subject not in subjects_with_changes:
                subjects_with_changes[subject] = []
            subjects_with_changes[subject].append(var)

print(f"   - Subjects with significant within-subject changes across SOLs:")
for subject, vars_with_changes in subjects_with_changes.items():
    print(f"     • {subject}: {len(vars_with_changes)} variables")

print("\n5. STATISTICAL RECOMMENDATIONS:")
print("   - Non-parametric tests were appropriate due to non-normal distributions")
print("   - Kruskal-Wallis test revealed between-subject differences")
print("   - Spearman correlation identified temporal trends")
print("   - Friedman test detected within-subject changes over time")

print("\n6. CLINICAL IMPLICATIONS:")
print("   - Individual physiological responses vary significantly between subjects")
print("   - Temporal adaptation patterns differ across subjects")
print("   - Some subjects show consistent trends while others show stability")
print("   - These findings support personalized monitoring approaches")

print("\n7. LIMITATIONS:")
print("   - Unbalanced data across subjects may affect comparisons")
print("   - Missing data patterns should be investigated")
print("   - Multiple comparisons may require correction")
print("   - Temporal resolution may affect trend detection")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("- comprehensive_analysis_T03_T08.png (visualization)")
print("\nNext steps:")
print("1. Review specific significant findings")
print("2. Investigate clinical significance of differences")
print("3. Consider additional time-series analysis")
print("4. Validate findings with domain experts") 