# =============================================================================
# TIME ANALYSIS OF T01_MARA AND T02_LAURA PHYSIOLOGICAL DATA ACROSS SOLS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("TIME ANALYSIS: T01_MARA vs T02_LAURA ACROSS SOLS")
print("="*80)

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

print("\n1. LOADING AND PREPARING DATA...")
print("-" * 50)

# Load data
file_paths = {
    'T01_Mara': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T01_Mara.csv',
    'T02_Laura': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T02_Laura.csv'
}

dfs = {}
for name, path in file_paths.items():
    try:
        dfs[name] = pd.read_csv(path)
        print(f"✓ Loaded {name}: {dfs[name].shape[0]:,} rows")
    except Exception as e:
        print(f"✗ Error loading {name}: {e}")

# Combine and prepare data
if len(dfs) == 2:
    # Add source identifier
    dfs['T01_Mara']['source'] = 'T01_Mara'
    dfs['T02_Laura']['source'] = 'T02_Laura'
    
    # Combine dataframes
    df_combined = pd.concat([dfs['T01_Mara'], dfs['T02_Laura']], ignore_index=True)
    
    # Convert time to datetime
    df_combined['datetime'] = pd.to_datetime(df_combined['time [s/1000]'], unit='ms')
    
    # Extract date components
    df_combined['date'] = df_combined['datetime'].dt.date
    df_combined['hour'] = df_combined['datetime'].dt.hour
    df_combined['day_of_week'] = df_combined['datetime'].dt.day_name()
    
    print(f"✓ Combined dataset: {df_combined.shape[0]:,} rows")
    print(f"✓ Time range: {df_combined['datetime'].min()} to {df_combined['datetime'].max()}")
    print(f"✓ Sol range: {df_combined['Sol'].min()} to {df_combined['Sol'].max()}")
else:
    print("✗ Could not load data")
    df_combined = None

# =============================================================================
# 2. SOL-BY-SOL ANALYSIS
# =============================================================================

print("\n2. SOL-BY-SOL ANALYSIS")
print("-" * 50)

if df_combined is not None:
    # Key physiological variables to analyze
    key_vars = ['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]', 'SPO2 [%]', 'minute_ventilation [mL/min]']
    available_vars = [var for var in key_vars if var in df_combined.columns]
    
    print(f"Analyzing {len(available_vars)} variables across Sols:")
    for var in available_vars:
        print(f"  - {var}")
    
    # Calculate daily statistics for each variable and subject
    daily_stats = {}
    
    for var in available_vars:
        print(f"\nAnalyzing {var}...")
        
        # Group by Sol and source, calculate statistics
        stats_by_sol = df_combined.groupby(['Sol', 'source'])[var].agg([
            'count', 'mean', 'std', 'median', 'min', 'max',
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75)   # Q3
        ]).round(3)
        
        # Rename columns
        stats_by_sol.columns = ['count', 'mean', 'std', 'median', 'min', 'max', 'q25', 'q75']
        
        # Reset index for easier access
        stats_by_sol = stats_by_sol.reset_index()
        
        daily_stats[var] = stats_by_sol
        
        # Print summary for each subject
        for subject in ['T01_Mara', 'T02_Laura']:
            subject_data = stats_by_sol[stats_by_sol['source'] == subject]
            if len(subject_data) > 0:
                print(f"  {subject}: {len(subject_data)} Sols, Mean across Sols: {subject_data['mean'].mean():.2f}")

# =============================================================================
# 3. VISUALIZATION: SOL-BY-SOL TRENDS
# =============================================================================

print("\n3. CREATING SOL-BY-SOL VISUALIZATIONS")
print("-" * 50)

if df_combined is not None and len(available_vars) > 0:
    
    # Plot 1: Time series of key variables across Sols
    fig, axes = plt.subplots(len(available_vars), 1, figsize=(15, 4*len(available_vars)))
    if len(available_vars) == 1:
        axes = [axes]
    
    for i, var in enumerate(available_vars):
        # Get daily means for each subject
        daily_means = df_combined.groupby(['Sol', 'source'])[var].mean().reset_index()
        
        # Plot for each subject
        for subject in ['T01_Mara', 'T02_Laura']:
            subject_data = daily_means[daily_means['source'] == subject]
            if len(subject_data) > 0:
                axes[i].plot(subject_data['Sol'], subject_data[var], 
                           marker='o', linewidth=2, markersize=6, 
                           label=subject, alpha=0.8)
        
        axes[i].set_title(f'{var} - Daily Mean Across Sols', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Sol (Day)', fontsize=12)
        axes[i].set_ylabel(var, fontsize=12)
        axes[i].legend(fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Box plots comparing distributions across Sols
    key_vars_for_box = ['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]']
    available_box_vars = [var for var in key_vars_for_box if var in df_combined.columns]
    
    if available_box_vars:
        fig, axes = plt.subplots(1, len(available_box_vars), figsize=(6*len(available_box_vars), 8))
        if len(available_box_vars) == 1:
            axes = [axes]
        
        for i, var in enumerate(available_box_vars):
            # Create box plot by Sol and source
            sns.boxplot(data=df_combined, x='Sol', y=var, hue='source', ax=axes[i])
            axes[i].set_title(f'{var} Distribution by Sol', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Sol (Day)', fontsize=12)
            axes[i].set_ylabel(var, fontsize=12)
            axes[i].legend(title='Subject', fontsize=11)
            axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# 4. STATISTICAL COMPARISON BETWEEN SUBJECTS
# =============================================================================

print("\n4. STATISTICAL COMPARISON BETWEEN SUBJECTS")
print("-" * 50)

if df_combined is not None and len(available_vars) > 0:
    
    print("COMPARING T01_MARA vs T02_LAURA:")
    print("="*60)
    
    comparison_results = {}
    
    for var in available_vars:
        print(f"\n{var}:")
        print("-" * 30)
        
        # Get data for each subject
        t01_data = df_combined[df_combined['source'] == 'T01_Mara'][var].dropna()
        t02_data = df_combined[df_combined['source'] == 'T02_Laura'][var].dropna()
        
        if len(t01_data) > 0 and len(t02_data) > 0:
            # Descriptive statistics
            print(f"T01_Mara:  n={len(t01_data):,}, mean={t01_data.mean():.3f}, std={t01_data.std():.3f}")
            print(f"T02_Laura: n={len(t02_data):,}, mean={t02_data.mean():.3f}, std={t02_data.std():.3f}")
            
            # Mann-Whitney U test (non-parametric, appropriate for non-normal data)
            try:
                stat, p_value = stats.mannwhitneyu(t01_data, t02_data, alternative='two-sided')
                print(f"Mann-Whitney U test: U={stat:.0f}, p={p_value:.2e}")
                
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                print(f"Significance: {significance}")
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(t01_data) - 1) * t01_data.var() + (len(t02_data) - 1) * t02_data.var()) / 
                                   (len(t01_data) + len(t02_data) - 2))
                cohens_d = (t01_data.mean() - t02_data.mean()) / pooled_std
                print(f"Effect size (Cohen's d): {cohens_d:.3f}")
                
                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    effect_interpretation = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_interpretation = "small"
                elif abs(cohens_d) < 0.8:
                    effect_interpretation = "medium"
                else:
                    effect_interpretation = "large"
                print(f"Effect size interpretation: {effect_interpretation}")
                
                comparison_results[var] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significance': significance,
                    'cohens_d': cohens_d,
                    'effect_size': effect_interpretation
                }
                
            except Exception as e:
                print(f"Error in statistical test: {e}")
        else:
            print("Insufficient data for comparison")

# =============================================================================
# 5. DAILY PATTERNS ANALYSIS
# =============================================================================

print("\n5. DAILY PATTERNS ANALYSIS")
print("-" * 50)

if df_combined is not None and len(available_vars) > 0:
    
    # Analyze hourly patterns
    print("Analyzing hourly patterns...")
    
    # Create hourly averages for each subject
    hourly_patterns = {}
    
    for var in available_vars:
        hourly_data = df_combined.groupby(['hour', 'source'])[var].mean().reset_index()
        hourly_patterns[var] = hourly_data
    
    # Plot hourly patterns
    fig, axes = plt.subplots(len(available_vars), 1, figsize=(15, 4*len(available_vars)))
    if len(available_vars) == 1:
        axes = [axes]
    
    for i, var in enumerate(available_vars):
        hourly_data = hourly_patterns[var]
        
        for subject in ['T01_Mara', 'T02_Laura']:
            subject_data = hourly_data[hourly_data['source'] == subject]
            if len(subject_data) > 0:
                axes[i].plot(subject_data['hour'], subject_data[var], 
                           marker='o', linewidth=2, markersize=6, 
                           label=subject, alpha=0.8)
        
        axes[i].set_title(f'{var} - Hourly Patterns (24-hour cycle)', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Hour of Day', fontsize=12)
        axes[i].set_ylabel(var, fontsize=12)
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].legend(fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. CORRELATION ANALYSIS ACROSS TIME
# =============================================================================

print("\n6. CORRELATION ANALYSIS ACROSS TIME")
print("-" * 50)

if df_combined is not None and len(available_vars) > 0:
    
    print("Analyzing correlations between variables across time...")
    
    # Select key variables for correlation
    corr_vars = ['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]', 'SPO2 [%]']
    available_corr_vars = [var for var in corr_vars if var in df_combined.columns]
    
    if len(available_corr_vars) >= 2:
        # Calculate correlations for each subject
        for subject in ['T01_Mara', 'T02_Laura']:
            print(f"\n{subject} - Spearman Correlations:")
            print("-" * 40)
            
            subject_data = df_combined[df_combined['source'] == subject][available_corr_vars].dropna()
            
            if len(subject_data) > 10:  # Need sufficient data
                corr_matrix = subject_data.corr(method='spearman')
                
                # Print correlation matrix
                print("Correlation Matrix:")
                print(corr_matrix.round(3))
                
                # Find strongest correlations
                print("\nStrongest correlations (|r| > 0.3):")
                for i in range(len(available_corr_vars)):
                    for j in range(i+1, len(available_corr_vars)):
                        var1 = available_corr_vars[i]
                        var2 = available_corr_vars[j]
                        corr_value = corr_matrix.loc[var1, var2]
                        
                        if abs(corr_value) > 0.3:
                            direction = "positive" if corr_value > 0 else "negative"
                            strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.5 else "weak"
                            print(f"  {var1} ↔ {var2}: r = {corr_value:.3f} ({strength} {direction})")
            else:
                print("Insufficient data for correlation analysis")

# =============================================================================
# 7. SUMMARY STATISTICS BY SOL
# =============================================================================

print("\n7. SUMMARY STATISTICS BY SOL")
print("-" * 50)

if df_combined is not None and len(available_vars) > 0:
    
    # Create summary table
    summary_data = []
    
    for var in available_vars:
        for subject in ['T01_Mara', 'T02_Laura']:
            subject_data = df_combined[df_combined['source'] == subject]
            
            # Get Sols for this subject
            sols = sorted(subject_data['Sol'].unique())
            
            for sol in sols:
                sol_data = subject_data[subject_data['Sol'] == sol][var].dropna()
                
                if len(sol_data) > 0:
                    summary_data.append({
                        'Variable': var,
                        'Subject': subject,
                        'Sol': sol,
                        'Count': len(sol_data),
                        'Mean': sol_data.mean(),
                        'Std': sol_data.std(),
                        'Median': sol_data.median(),
                        'Min': sol_data.min(),
                        'Max': sol_data.max()
                    })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display summary for key variables
    key_summary_vars = ['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]']
    
    for var in key_summary_vars:
        if var in summary_df['Variable'].values:
            print(f"\n{var} - Summary by Sol and Subject:")
            print("-" * 50)
            
            var_summary = summary_df[summary_df['Variable'] == var].round(3)
            
            # Pivot table for easier reading
            pivot_summary = var_summary.pivot_table(
                values=['Mean', 'Std', 'Count'], 
                index='Sol', 
                columns='Subject', 
                aggfunc='first'
            )
            
            print(pivot_summary)

# =============================================================================
# 8. FINAL SUMMARY REPORT
# =============================================================================

print("\n8. FINAL SUMMARY REPORT")
print("-" * 50)

if df_combined is not None:
    print("TEMPORAL ANALYSIS SUMMARY:")
    print("="*60)
    
    # Dataset overview
    print(f"• Total observations: {len(df_combined):,}")
    print(f"• Time span: {df_combined['Sol'].min()} to {df_combined['Sol'].max()} Sols")
    print(f"• Subjects: {df_combined['source'].unique()}")
    
    # Data availability by subject
    print("\nData availability by subject:")
    for subject in ['T01_Mara', 'T02_Laura']:
        subject_data = df_combined[df_combined['source'] == subject]
        print(f"  {subject}: {len(subject_data):,} observations across {len(subject_data['Sol'].unique())} Sols")
    
    # Key findings from statistical comparisons
    if 'comparison_results' in locals():
        print("\nStatistical comparison results:")
        for var, result in comparison_results.items():
            print(f"  {var}: {result['significance']} (p={result['p_value']:.2e}, d={result['cohens_d']:.3f})")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("• Use non-parametric tests for statistical comparisons (data is non-normal)")
    print("• Consider individual differences when analyzing physiological patterns")
    print("• Account for temporal autocorrelation in time-series analyses")
    print("• Monitor data quality and missing patterns across Sols")

print("\n" + "="*80)
print("TIME ANALYSIS COMPLETE")
print("="*80) 