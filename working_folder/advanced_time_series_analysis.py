import pandas as pd
import statsmodels.formula.api as smf
from pygam import GAM, s, f
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# --- Helper Functions and Setup ---

def load_and_prepare_data(data_path='hrv_results/hrv_complete.csv'):
    """Loads and prepares the HRV data for advanced analysis."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the data file
        full_data_path = os.path.join(script_dir, data_path)
        
        data = pd.read_csv(full_data_path)
        # Clean data (similar to your original script)
        data = data.dropna(subset=['Subject', 'Sol'])
        data['Subject'] = data['Subject'].astype(str)
        data['Sol'] = data['Sol'].astype(int)
        
        # For pygam, we need numerical codes for subjects to use as factors
        data['Subject_code'] = data['Subject'].astype('category').cat.codes
        
        print(f"✓ Data loaded successfully from {full_data_path}")
        print(f"✓ Data shape: {data.shape}")
        print(f"✓ Subjects: {data['Subject'].unique()}")
        print(f"✓ Sol range: {data['Sol'].min()} - {data['Sol'].max()}")
        return data
    except FileNotFoundError:
        print(f"✗ Error: Data file not found at {full_data_path}")
        print(f"✗ Current working directory: {os.getcwd()}")
        print(f"✗ Script directory: {script_dir}")
        return None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

# Define the metrics to analyze, same as in your parasympathetic_time_analysis.py
parasympathetic_metrics = ['rmssd', 'pnni_50', 'pnni_20', 'hf', 'hfnu', 'sd1']

# --- 1. Mixed-Effects Models (MEMs) Analysis ---

def run_mixed_effects_models(data, metrics):
    """
    Fits and summarizes a Mixed-Effects Model for each specified metric.
    This model assesses the overall trend ('fixed effect') of the mission day (Sol)
    while accounting for individual crew member differences ('random effects').
    """
    print("\n" + "="*60)
    print(" Recommendation 1: Running Mixed-Effects Models (MEMs)")
    print("="*60)
    
    for metric in metrics:
        if metric not in data.columns:
            print(f"\n--- Skipping {metric.upper()}: Column not found ---")
            continue
            
        print(f"\n--- Analyzing Metric: {metric.upper()} ---")
        
        model_formula = f"{metric} ~ Sol"
        
        try:
            model_data = data[['Sol', 'Subject', metric]].dropna()

            if len(model_data['Subject'].unique()) < 2 or len(model_data) < 15:
                print("Not enough data to robustly fit the model.")
                continue

            mem = smf.mixedlm(model_formula, model_data, groups=model_data["Subject"], re_formula="~Sol")
            result = mem.fit(method='powell')
            
            print(result.summary())
            print("\nInterpretation:")
            p_value = result.pvalues['Sol']
            coef = result.params['Sol']
            interpretation = "a significant" if p_value < 0.05 else "no significant"
            direction = "increase" if coef > 0 else "decrease"
            print(f"The fixed effect for 'Sol' (P-value: {p_value:.4f}) indicates that, on average, there is {interpretation} {direction} in {metric.upper()} per day across the mission.")

        except Exception as e:
            print(f"Could not fit model for {metric}. Error: {e}")

# --- 2. Generalized Additive Models (GAMs) Analysis ---

def run_generalized_additive_models(data, metrics):
    """
    Fits and summarizes a Generalized Additive Model for each specified metric
    to capture potentially complex, non-linear trends over time.
    """
    print("\n" + "="*60)
    print(" Recommendation 2: Running Generalized Additive Models (GAMs)")
    print("="*60)
    
    for metric in metrics:
        if metric not in data.columns:
            print(f"\n--- Skipping {metric.upper()}: Column not found ---")
            continue
            
        print(f"\n--- Analyzing Metric: {metric.upper()} ---")
        
        try:
            model_data = data[['Sol', 'Subject', 'Subject_code', metric]].dropna()

            if len(model_data['Subject_code'].unique()) < 2 or len(model_data) < 15:
                print("Not enough data to robustly fit the model.")
                continue

            X = model_data[['Sol', 'Subject_code']]
            y = model_data[metric]

            gam = GAM(s(0, n_splines=10) + f(1)).fit(X, y)
            
            print(gam.summary())
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            XX = gam.generate_X_grid(term=0)
            pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
            
            ax.plot(XX[:, 0], pdep, color='royalblue', linewidth=3)
            ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], color='cornflowerblue', alpha=0.3, label='95% Confidence Interval')
            sns.scatterplot(x='Sol', y=metric, hue='Subject', data=model_data, ax=ax, alpha=0.6, palette='husl')
            
            ax.set_title(f"Non-Linear Trend for {metric.upper()} (GAM Analysis)", fontsize=16)
            ax.set_xlabel("Sol (Mission Day)", fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, f'{metric}_gam_trend.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"\n✓ GAM plot for {metric.upper()} saved to: {save_path}")

        except AttributeError as e:
            if "'numpy' has no attribute 'int'" in str(e):
                print("\n✗ ERROR: pyGAM and NumPy version conflict.")
                print("  This is a known issue with older versions of pyGAM and newer versions of NumPy.")
                print("  Please update pyGAM to the latest version by running this command in your terminal:")
                print("  pip install --upgrade pygam")
            else:
                print(f"Could not fit model for {metric}. An AttributeError occurred: {e}")
        except Exception as e:
            print(f"Could not fit model for {metric}. An unexpected error occurred: {e}")

# --- Main Execution ---

if __name__ == '__main__':
    hrv_data = load_and_prepare_data()
    
    if hrv_data is not None:
        run_mixed_effects_models(hrv_data, parasympathetic_metrics)
        run_generalized_additive_models(hrv_data, parasympathetic_metrics) 