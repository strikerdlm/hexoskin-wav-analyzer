import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set the figure style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Load the dataset
file_path = r'C:\Users\User\OneDrive\FAC\Research\Valquiria\Data\full_physiological_dataset.csv'
df = pd.read_csv(file_path)

# Basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
for col, count, pct in zip(missing_counts.index, missing_counts, missing_percent):
    print(f"{col}: {count} ({pct:.1f}%)")

# Check if 'sol' column exists for time analysis
if 'sol' in df.columns:
    print("\nSol column found - proceeding with time analysis")
    sol_col = 'sol'
elif any('sol' in col.lower() for col in df.columns):
    # Find columns that might contain sol information
    sol_candidates = [col for col in df.columns if 'sol' in col.lower()]
    print(f"\nFound potential sol columns: {sol_candidates}")
    sol_col = sol_candidates[0]  # Using the first candidate
else:
    # Try to find date/time columns
    date_candidates = [col for col in df.columns if any(term in col.lower() 
                      for term in ['date', 'time', 'day', 'sol'])]
    
    if date_candidates:
        print(f"\nNo sol column found, but found these potential time columns: {date_candidates}")
        sol_col = date_candidates[0]
    else:
        print("\nNo time/sol columns found. Cannot perform time analysis.")
        sol_col = None

# Time analysis by sols (if sol column exists)
if sol_col:
    print(f"\nPerforming time analysis using '{sol_col}' column")
    
    # Check if the column needs to be converted to datetime
    if df[sol_col].dtype == 'object':
        try:
            df['date_time'] = pd.to_datetime(df[sol_col])
            print(f"Converted '{sol_col}' to datetime")
            time_col = 'date_time'
        except:
            print(f"Could not convert '{sol_col}' to datetime. Using as is.")
            time_col = sol_col
    else:
        time_col = sol_col
    
    # Group by the time column
    if 'date_time' in df.columns:
        # If we converted to datetime, we can group by day
        df['date'] = df['date_time'].dt.date
        grouped = df.groupby('date')
        print("\nData points per day:")
        print(grouped.size())
    else:
        # Otherwise group by the original column
        grouped = df.groupby(time_col)
        print(f"\nData points per {time_col}:")
        print(grouped.size())

    # Identify key health metrics for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out non-physiological metrics (like IDs, indices, etc.)
    health_metrics = [col for col in numeric_cols 
                     if any(term in col.lower() for term in 
                           ['rate', 'pressure', 'temp', 'spo2', 'heart', 'breath', 
                            'bpm', 'mmhg', 'celsius', 'fahrenheit'])]
    
    if not health_metrics:
        # If no clear health metrics, use all numeric columns except the time column
        health_metrics = [col for col in numeric_cols if col != time_col]
    
    print(f"\nAnalyzing these health metrics over time: {health_metrics}")
    
    # Time series plots for key metrics
    for metric in health_metrics[:5]:  # Limit to first 5 metrics to avoid too many plots
        plt.figure(figsize=(12, 6))
        if 'date' in df.columns:
            # Plot by date if available
            df.groupby('date')[metric].mean().plot()
            plt.title(f'Average {metric} by Date')
            plt.xlabel('Date')
        else:
            # Otherwise plot by the time column
            df.groupby(time_col)[metric].mean().plot()
            plt.title(f'Average {metric} by {time_col}')
            plt.xlabel(time_col)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'time_analysis_{metric.replace(" ", "_").replace("[", "").replace("]", "")}.png')
        plt.close()

# Normality analysis
print("\n\nNormality Analysis")
print("=================")

# Identify numeric columns for normality testing
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if sol_col and sol_col in numeric_cols:
    numeric_cols.remove(sol_col)  # Remove time column from analysis

normality_results = {}
for col in numeric_cols:
    # Skip columns with too many missing values
    if df[col].isnull().sum() / len(df) > 0.5:
        print(f"{col}: Too many missing values (>50%), skipping normality test")
        continue
        
    # Get non-null values
    data = df[col].dropna()
    
    if len(data) < 8:
        print(f"{col}: Not enough data points for normality test")
        continue
        
    # Shapiro-Wilk test (best for small samples)
    if len(data) < 5000:
        stat, p = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        # For larger samples, use D'Agostino-Pearson test
        stat, p = stats.normaltest(data)
        test_name = "D'Agostino-Pearson"
    
    alpha = 0.05
    normality = p > alpha
    
    print(f"{col}: {test_name} test p-value = {p:.6f}, {'Normal' if normality else 'Not normal'}")
    
    # Save result
    normality_results[col] = {
        'test': test_name,
        'p_value': p,
        'is_normal': normality
    }
    
    # Create QQ plot and histogram to visually check normality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram with KDE
    sns.histplot(data, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {col}')
    
    # QQ plot
    stats.probplot(data, plot=ax2)
    ax2.set_title(f'Q-Q Plot of {col}')
    
    plt.tight_layout()
    plt.savefig(f'normality_{col.replace(" ", "_").replace("[", "").replace("]", "")}.png')
    plt.close()

# Recommend statistical methods based on normality results
print("\n\nRecommended Statistical Methods")
print("==============================")

if not normality_results:
    print("Could not determine normality for any variables.")
else:
    # Count how many variables are normal vs non-normal
    normal_vars = sum(1 for res in normality_results.values() if res['is_normal'])
    non_normal_vars = len(normality_results) - normal_vars
    
    print(f"Normal variables: {normal_vars}")
    print(f"Non-normal variables: {non_normal_vars}")
    
    if normal_vars > non_normal_vars:
        print("\nMost variables appear normally distributed.")
        print("Recommended statistical methods:")
        print("1. Parametric tests like t-tests for comparing two groups")
        print("2. ANOVA for comparing multiple groups")
        print("3. Pearson correlation for examining relationships between variables")
        print("4. Linear regression for predictive modeling")
    else:
        print("\nMost variables appear non-normally distributed.")
        print("Recommended statistical methods:")
        print("1. Non-parametric tests like Mann-Whitney U for comparing two groups")
        print("2. Kruskal-Wallis test for comparing multiple groups")
        print("3. Spearman correlation for examining relationships between variables")
        print("4. Consider data transformations (log, sqrt) before using parametric tests")
        print("5. Or robust regression methods that don't assume normality")

    # If time series data is involved
    if sol_col:
        print("\nFor time series analysis:")
        if normal_vars > non_normal_vars:
            print("1. ARIMA or SARIMA models for forecasting")
            print("2. Linear mixed effects models for repeated measures")
        else:
            print("1. Non-parametric time series methods")
            print("2. Consider transformations before using traditional time series models")
            print("3. Quantile regression for handling skewed distributions")

print("\nAnalysis complete. Check the generated plots for visual examination of the data.") 