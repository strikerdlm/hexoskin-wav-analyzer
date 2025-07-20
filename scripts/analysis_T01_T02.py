# =============================================================================
# COMPREHENSIVE ANALYSIS OF T01_MARA AND T02_LAURA DATA
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("ANALYSIS OF T01_MARA AND T02_LAURA PHYSIOLOGICAL DATA")
print("="*80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("\n1. LOADING DATA FILES...")
print("-" * 50)

# Define file paths
file_paths = {
    'T01_Mara': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T01_Mara.csv',
    'T02_Laura': r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\joined_data\T02_Laura.csv'
}

# Load data
dfs = {}
for name, path in file_paths.items():
    try:
        dfs[name] = pd.read_csv(path)
        print(f"✓ Successfully loaded {name}: {dfs[name].shape[0]:,} rows, {dfs[name].shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading {name}: {e}")

# Combine dataframes
if len(dfs) == 2:
    df_combined = pd.concat([dfs['T01_Mara'], dfs['T02_Laura']], ignore_index=True)
    print(f"✓ Combined dataset: {df_combined.shape[0]:,} rows, {df_combined.shape[1]} columns")
else:
    print("✗ Could not combine datasets - check file loading")
    df_combined = None

# =============================================================================
# 2. VARIABLE LISTING AND OVERVIEW
# =============================================================================

print("\n2. VARIABLE OVERVIEW")
print("-" * 50)

if df_combined is not None:
    print("Variables in the dataset:")
    print("-" * 30)
    
    for i, col in enumerate(df_combined.columns, 1):
        dtype = df_combined[col].dtype
        missing = df_combined[col].isnull().sum()
        missing_pct = (missing / len(df_combined)) * 100
        
        print(f"{i:2d}. {col:<25} | Type: {str(dtype):<10} | Missing: {missing:>6} ({missing_pct:5.1f}%)")
    
    print(f"\nTotal variables: {len(df_combined.columns)}")
    print(f"Total observations: {len(df_combined):,}")

# =============================================================================
# 3. DESCRIPTIVE ANALYSIS
# =============================================================================

print("\n3. DESCRIPTIVE ANALYSIS")
print("-" * 50)

if df_combined is not None:
    # Basic info
    print("Dataset Information:")
    print(df_combined.info())
    
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    
    # Descriptive statistics for numerical columns
    numerical_cols = df_combined.select_dtypes(include=[np.number]).columns
    categorical_cols = df_combined.select_dtypes(include=['object']).columns
    
    print(f"\nNumerical Variables ({len(numerical_cols)}):")
    print("-" * 40)
    desc_stats = df_combined[numerical_cols].describe()
    print(desc_stats)
    
    print(f"\nCategorical Variables ({len(categorical_cols)}):")
    print("-" * 40)
    for col in categorical_cols:
        print(f"\n{col}:")
        value_counts = df_combined[col].value_counts()
        print(value_counts.head(10))  # Show top 10 values
        if len(value_counts) > 10:
            print(f"... and {len(value_counts) - 10} more unique values")

# =============================================================================
# 4. NORMALITY ANALYSIS
# =============================================================================

print("\n4. NORMALITY ANALYSIS")
print("-" * 50)

def normality_test_comprehensive(data, variable_name):
    """
    Perform comprehensive normality tests on a variable
    """
    # Remove NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        return f"Not enough data for {variable_name} (n < 3)"
    
    results = {
        'variable': variable_name,
        'n': len(clean_data),
        'mean': np.mean(clean_data),
        'std': np.std(clean_data),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data),
        'shapiro_w': None,
        'shapiro_p': None,
        'normaltest_stat': None,
        'normaltest_p': None,
        'anderson_stat': None,
        'anderson_critical': None
    }
    
    # Shapiro-Wilk test (recommended for n < 50)
    if len(clean_data) <= 5000:  # Shapiro-Wilk has limitations for large samples
        try:
            shapiro_stat, shapiro_p = shapiro(clean_data)
            results['shapiro_w'] = shapiro_stat
            results['shapiro_p'] = shapiro_p
        except:
            pass
    
    # D'Agostino K^2 test (good for larger samples)
    try:
        normaltest_stat, normaltest_p = normaltest(clean_data)
        results['normaltest_stat'] = normaltest_stat
        results['normaltest_p'] = normaltest_p
    except:
        pass
    
    # Anderson-Darling test
    try:
        anderson_result = anderson(clean_data)
        results['anderson_stat'] = anderson_result.statistic
        # Get critical value at 5% significance level
        results['anderson_critical'] = anderson_result.critical_values[2]  # 5% level
    except:
        pass
    
    return results

if df_combined is not None:
    # Select numerical variables for normality testing
    numerical_vars = df_combined.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove variables that are likely not continuous (like Sol, time)
    exclude_vars = ['Sol', 'time [s/1000]']
    test_vars = [var for var in numerical_vars if var not in exclude_vars]
    
    print(f"Testing normality for {len(test_vars)} numerical variables:")
    print("="*80)
    
    normality_results = []
    
    for var in test_vars:
        print(f"\nTesting: {var}")
        print("-" * 40)
        
        result = normality_test_comprehensive(df_combined[var], var)
        
        if isinstance(result, str):
            print(result)
        else:
            normality_results.append(result)
            
            print(f"Sample size: {result['n']:,}")
            print(f"Mean: {result['mean']:.4f}")
            print(f"Std: {result['std']:.4f}")
            print(f"Skewness: {result['skewness']:.4f}")
            print(f"Kurtosis: {result['kurtosis']:.4f}")
            
            # Interpret skewness and kurtosis
            if abs(result['skewness']) > 1:
                print(f"  → Highly skewed ({'positive' if result['skewness'] > 0 else 'negative'})")
            elif abs(result['skewness']) > 0.5:
                print(f"  → Moderately skewed ({'positive' if result['skewness'] > 0 else 'negative'})")
            else:
                print(f"  → Approximately symmetric")
                
            if abs(result['kurtosis']) > 3:
                print(f"  → Heavy tails (leptokurtic)")
            elif abs(result['kurtosis']) < -1:
                print(f"  → Light tails (platykurtic)")
            else:
                print(f"  → Normal-like kurtosis")
            
            # Test results
            if result['shapiro_p'] is not None:
                print(f"Shapiro-Wilk: W = {result['shapiro_w']:.4f}, p = {result['shapiro_p']:.4e}")
                if result['shapiro_p'] < 0.05:
                    print("  → NOT normal (p < 0.05)")
                else:
                    print("  → Normal (p ≥ 0.05)")
            
            if result['normaltest_p'] is not None:
                print(f"D'Agostino K²: χ² = {result['normaltest_stat']:.4f}, p = {result['normaltest_p']:.4e}")
                if result['normaltest_p'] < 0.05:
                    print("  → NOT normal (p < 0.05)")
                else:
                    print("  → Normal (p ≥ 0.05)")
            
            if result['anderson_stat'] is not None:
                print(f"Anderson-Darling: A² = {result['anderson_stat']:.4f}, Critical (5%) = {result['anderson_critical']:.4f}")
                if result['anderson_stat'] > result['anderson_critical']:
                    print("  → NOT normal (A² > critical)")
                else:
                    print("  → Normal (A² ≤ critical)")

# =============================================================================
# 5. VISUALIZATION
# =============================================================================

print("\n5. CREATING VISUALIZATIONS")
print("-" * 50)

if df_combined is not None and len(normality_results) > 0:
    
    # Create a summary table of normality results
    summary_df = pd.DataFrame(normality_results)
    
    # Plot 1: Distribution plots for key variables
    key_vars = ['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]', 'SPO2 [%]']
    available_vars = [var for var in key_vars if var in df_combined.columns]
    
    if available_vars:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, var in enumerate(available_vars):
            if i < 4:  # Limit to 4 plots
                # Remove NaN values
                clean_data = df_combined[var].dropna()
                
                if len(clean_data) > 0:
                    # Histogram with normal curve
                    axes[i].hist(clean_data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Add normal curve
                    x = np.linspace(clean_data.min(), clean_data.max(), 100)
                    normal_curve = stats.norm.pdf(x, clean_data.mean(), clean_data.std())
                    axes[i].plot(x, normal_curve, 'r-', linewidth=2, label='Normal distribution')
                    
                    axes[i].set_title(f'Distribution of {var}\nSkewness: {stats.skew(clean_data):.3f}, Kurtosis: {stats.kurtosis(clean_data):.3f}')
                    axes[i].set_xlabel(var)
                    axes[i].set_ylabel('Density')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Plot 2: Q-Q plots for normality assessment
    if available_vars:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, var in enumerate(available_vars):
            if i < 4:
                clean_data = df_combined[var].dropna()
                
                if len(clean_data) > 0:
                    stats.probplot(clean_data, dist="norm", plot=axes[i])
                    axes[i].set_title(f'Q-Q Plot: {var}')
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Plot 3: Box plots by user
    if 'user' in df_combined.columns:
        key_vars_for_box = ['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]']
        available_box_vars = [var for var in key_vars_for_box if var in df_combined.columns]
        
        if available_box_vars:
            fig, axes = plt.subplots(1, len(available_box_vars), figsize=(5*len(available_box_vars), 6))
            if len(available_box_vars) == 1:
                axes = [axes]
            
            for i, var in enumerate(available_box_vars):
                df_combined.boxplot(column=var, by='user', ax=axes[i])
                axes[i].set_title(f'{var} by User')
                axes[i].set_xlabel('User')
                axes[i].set_ylabel(var)
            
            plt.tight_layout()
            plt.show()

# =============================================================================
# 6. SUMMARY REPORT
# =============================================================================

print("\n6. SUMMARY REPORT")
print("-" * 50)

if df_combined is not None:
    print("DATASET SUMMARY:")
    print(f"• Total observations: {len(df_combined):,}")
    print(f"• Total variables: {len(df_combined.columns)}")
    print(f"• Users: {df_combined['user'].unique()}")
    print(f"• Date range: {df_combined['Sol'].min()} to {df_combined['Sol'].max()} Sol")
    
    print("\nKEY FINDINGS:")
    
    # Missing data summary
    missing_summary = df_combined.isnull().sum()
    high_missing = missing_summary[missing_summary > len(df_combined) * 0.5]
    if len(high_missing) > 0:
        print(f"• Variables with >50% missing data: {list(high_missing.index)}")
    
    # Normality summary
    if len(normality_results) > 0:
        normal_vars = []
        non_normal_vars = []
        
        for result in normality_results:
            # Use multiple tests to determine normality
            tests_passed = 0
            total_tests = 0
            
            if result['shapiro_p'] is not None:
                total_tests += 1
                if result['shapiro_p'] >= 0.05:
                    tests_passed += 1
            
            if result['normaltest_p'] is not None:
                total_tests += 1
                if result['normaltest_p'] >= 0.05:
                    tests_passed += 1
            
            if result['anderson_stat'] is not None:
                total_tests += 1
                if result['anderson_stat'] <= result['anderson_critical']:
                    tests_passed += 1
            
            # Consider normal if majority of tests pass
            if total_tests > 0 and tests_passed >= total_tests / 2:
                normal_vars.append(result['variable'])
            else:
                non_normal_vars.append(result['variable'])
        
        print(f"• Variables that appear normal: {len(normal_vars)}")
        print(f"• Variables that are NOT normal: {len(non_normal_vars)}")
        
        if non_normal_vars:
            print(f"  Non-normal variables: {non_normal_vars}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80) 