"""
Analysis Utilities for Valquiria Data
Common functions for statistical analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kruskal, f_oneway
import warnings
warnings.filterwarnings('ignore')

def setup_plotting_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.dpi'] = 100

def analyze_variable_distribution(df, column, subject_col='subject'):
    """Analyze distribution of a variable across subjects."""
    setup_plotting_style()
    
    if column not in df.columns:
        print(f"Column '{column}' not found in dataset")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Distribution Analysis: {column}', fontsize=16)
    
    # Overall histogram
    data = df[column].dropna()
    axes[0, 0].hist(data, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Overall Distribution')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot by subject
    if subject_col in df.columns:
        df.boxplot(column=column, by=subject_col, ax=axes[0, 1])
        axes[0, 1].set_title('Distribution by Subject')
        axes[0, 1].set_xlabel('Subject')
        axes[0, 1].set_ylabel(column)
    else:
        axes[0, 1].text(0.5, 0.5, f'Subject column\n"{subject_col}" not found', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Distribution by Subject')
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    
    # Descriptive statistics
    stats_text = df[column].describe().to_string()
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Descriptive Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df, variables, method='pearson'):
    """Perform correlation analysis between variables."""
    # Filter variables that exist in the dataset
    available_vars = [var for var in variables if var in df.columns]
    if len(available_vars) < 2:
        print(f"Need at least 2 variables. Available: {available_vars}")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[available_vars].corr(method=method)
    
    # Create heatmap
    setup_plotting_style()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title(f'{method.title()} Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def time_series_analysis(df, variable, time_col='timestamp', subject_col='subject'):
    """Analyze time series patterns for a variable."""
    setup_plotting_style()
    
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in dataset")
        return
    
    if time_col not in df.columns:
        print(f"Time column '{time_col}' not found in dataset")
        return
    
    # Convert time column if needed
    if df[time_col].dtype == 'object':
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            print(f"Could not convert {time_col} to datetime")
            return
    
    # Plot time series by subject
    plt.figure(figsize=(15, 8))
    
    if subject_col in df.columns:
        for subject in df[subject_col].unique():
            subject_data = df[df[subject_col] == subject]
            plt.plot(subject_data[time_col], subject_data[variable], 
                    label=subject, alpha=0.7, linewidth=1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.plot(df[time_col], df[variable], alpha=0.7, linewidth=1)
    
    plt.title(f'Time Series: {variable}')
    plt.xlabel('Time')
    plt.ylabel(variable)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def statistical_comparison(df, variable, group_col='subject', test_type='kruskal'):
    """Perform statistical comparison between groups."""
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in dataset")
        return None, None
    
    if group_col not in df.columns:
        print(f"Group column '{group_col}' not found in dataset")
        return None, None
    
    groups = [group for name, group in df.groupby(group_col)]
    
    if test_type == 'kruskal':
        statistic, p_value = kruskal(*[group[variable].dropna() for group in groups])
        test_name = "Kruskal-Wallis H-test"
    elif test_type == 'anova':
        statistic, p_value = f_oneway(*[group[variable].dropna() for group in groups])
        test_name = "One-way ANOVA"
    else:
        raise ValueError("test_type must be 'kruskal' or 'anova'")
    
    print(f"{test_name} Results:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    return statistic, p_value

def quick_analysis(df, variables=['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]']):
    """Perform a quick analysis of key variables."""
    if df is None:
        print("No data provided for analysis")
        return
    
    print("="*60)
    print("QUICK ANALYSIS REPORT")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(f"Variables analyzed: {variables}")
    
    for var in variables:
        if var in df.columns:
            print(f"\n--- {var} ---")
            print(df[var].describe())
            
            # Distribution analysis
            analyze_variable_distribution(df, var)
            
            # Statistical comparison between subjects
            if 'subject' in df.columns:
                statistical_comparison(df, var)
        else:
            print(f"Variable {var} not found in dataset")

def normality_test(df, variable):
    """Perform normality tests on a variable."""
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in dataset")
        return
    
    data = df[variable].dropna()
    
    print(f"\nNormality Tests for {variable}")
    print("="*40)
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro-Wilk test:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  P-value: {shapiro_p:.4e}")
    print(f"  Normal: {'Yes' if shapiro_p > 0.05 else 'No'}")
    
    # Anderson-Darling test
    anderson_result = stats.anderson(data)
    print(f"\nAnderson-Darling test:")
    print(f"  Statistic: {anderson_result.statistic:.4f}")
    print(f"  Critical values: {anderson_result.critical_values}")
    print(f"  Significance levels: {anderson_result.significance_level}")
    
    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(data)
    print(f"\nJarque-Bera test:")
    print(f"  Statistic: {jb_stat:.4f}")
    print(f"  P-value: {jb_p:.4e}")
    print(f"  Normal: {'Yes' if jb_p > 0.05 else 'No'}")

def correlation_matrix_plot(df, variables=None, method='pearson'):
    """Create a correlation matrix plot with significance testing."""
    if variables is None:
        # Use numeric columns
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter variables that exist in the dataset
    available_vars = [var for var in variables if var in df.columns]
    if len(available_vars) < 2:
        print(f"Need at least 2 variables. Available: {available_vars}")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[available_vars].corr(method=method)
    
    # Calculate p-values for significance
    p_matrix = pd.DataFrame(index=available_vars, columns=available_vars)
    for i in available_vars:
        for j in available_vars:
            if i != j:
                data_i = df[i].dropna()
                data_j = df[j].dropna()
                # Align the data
                common_idx = data_i.index.intersection(data_j.index)
                if len(common_idx) > 2:
                    if method == 'pearson':
                        _, p_val = pearsonr(data_i.loc[common_idx], data_j.loc[common_idx])
                    else:  # spearman
                        _, p_val = spearmanr(data_i.loc[common_idx], data_j.loc[common_idx])
                    p_matrix.loc[i, j] = p_val
                else:
                    p_matrix.loc[i, j] = np.nan
            else:
                p_matrix.loc[i, j] = 1.0
    
    # Create the plot
    setup_plotting_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', ax=ax1)
    ax1.set_title(f'{method.title()} Correlation Matrix')
    
    # Significance heatmap
    significance_mask = p_matrix < 0.05
    sns.heatmap(significance_mask, annot=True, cmap='RdYlGn', 
                square=True, ax=ax2, cbar_kws={'label': 'Significant (p<0.05)'})
    ax2.set_title('Statistical Significance (p < 0.05)')
    
    plt.tight_layout()
    plt.show()
    
    return corr_matrix, p_matrix

def subject_comparison_plot(df, variable, subject_col='subject'):
    """Create comparison plots for a variable across subjects."""
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in dataset")
        return
    
    if subject_col not in df.columns:
        print(f"Subject column '{subject_col}' not found in dataset")
        return
    
    setup_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Subject Comparison: {variable}', fontsize=16)
    
    # Box plot
    df.boxplot(column=variable, by=subject_col, ax=axes[0, 0])
    axes[0, 0].set_title('Box Plot by Subject')
    axes[0, 0].set_xlabel('Subject')
    axes[0, 0].set_ylabel(variable)
    
    # Violin plot
    sns.violinplot(data=df, x=subject_col, y=variable, ax=axes[0, 1])
    axes[0, 1].set_title('Violin Plot by Subject')
    axes[0, 1].set_xlabel('Subject')
    axes[0, 1].set_ylabel(variable)
    
    # Histogram by subject
    for subject in df[subject_col].unique():
        subject_data = df[df[subject_col] == subject][variable].dropna()
        axes[1, 0].hist(subject_data, alpha=0.5, label=subject, bins=30)
    axes[1, 0].set_title('Histogram by Subject')
    axes[1, 0].set_xlabel(variable)
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Summary statistics
    summary_stats = df.groupby(subject_col)[variable].agg(['count', 'mean', 'std', 'min', 'max'])
    summary_text = summary_stats.to_string()
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Summary Statistics by Subject')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_plot(filename, dpi=300, bbox_inches='tight'):
    """Save the current plot with high quality."""
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved as: {filename}")

# Example usage functions
def run_comprehensive_analysis(df, variables=None):
    """Run a comprehensive analysis on the dataset."""
    if df is None:
        print("No data provided for analysis")
        return
    
    if variables is None:
        # Default variables for physiological data
        variables = ['heart_rate [bpm]', 'breathing_rate [rpm]', 'activity [g]', 'minute_ventilation [mL/min]']
    
    print("="*60)
    print("COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Quick overview
    quick_analysis(df, variables)
    
    # Correlation analysis
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    correlation_matrix_plot(df, variables)
    
    # Subject comparisons
    if 'subject' in df.columns:
        print("\n" + "="*60)
        print("SUBJECT COMPARISONS")
        print("="*60)
        for var in variables:
            if var in df.columns:
                subject_comparison_plot(df, var)

if __name__ == "__main__":
    print("Analysis utilities loaded successfully")
    print("Available functions:")
    print("- setup_plotting_style()")
    print("- analyze_variable_distribution(df, column)")
    print("- correlation_analysis(df, variables)")
    print("- time_series_analysis(df, variable)")
    print("- statistical_comparison(df, variable)")
    print("- quick_analysis(df, variables)")
    print("- normality_test(df, variable)")
    print("- correlation_matrix_plot(df, variables)")
    print("- subject_comparison_plot(df, variable)")
    print("- run_comprehensive_analysis(df, variables)") 