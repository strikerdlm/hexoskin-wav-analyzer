
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_data(db_path):
    """
    Performs a comprehensive analysis of the Hexoskin data,
    including data loading, cleaning, exploration, and visualization.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # Load the data into a pandas DataFrame
        df = pd.read_sql_query("SELECT * FROM hexoskin", conn)

        # --- 1. Data Cleaning and Preparation ---

        # Convert timestamps to a more readable format
        df['time'] = pd.to_datetime(df['time_seconds'], unit='s')

        # Handle missing values (if any)
        # For simplicity, we'll fill with the mean for now
        for col in ['breathing_rate [rpm]', 'heart_rate [bpm]', 'activity [g]']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)

        # --- 2. Exploratory Data Analysis (EDA) ---

        print("--- Data Overview ---")
        print(df.info())

        print("\n--- Summary Statistics ---")
        print(df.describe())

        # --- 3. Data Visualization ---

        print("\n--- Generating Visualizations ---")

        # Get the directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set the style for the plots
        sns.set_style("whitegrid")

        # a) Time series plot of heart rate
        plt.figure(figsize=(15, 6))
        sns.lineplot(x='time', y='heart_rate [bpm]', data=df, hue='subject')
        plt.title('Heart Rate Over Time')
        plt.savefig(os.path.join(script_dir, 'heart_rate_over_time.png'))
        plt.close()

        # b) Distribution of breathing rate
        plt.figure(figsize=(10, 6))
        sns.histplot(df['breathing_rate [rpm]'], kde=True, bins=30)
        plt.title('Distribution of Breathing Rate')
        plt.savefig(os.path.join(script_dir, 'breathing_rate_distribution.png'))
        plt.close()

        # c) Activity level vs. Heart rate
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='activity [g]', y='heart_rate [bpm]', data=df, alpha=0.5)
        plt.title('Activity Level vs. Heart Rate')
        plt.savefig(os.path.join(script_dir, 'activity_vs_heart_rate.png'))
        plt.close()

        print("\n--- Analysis Complete ---")
        print("Plots have been saved to the current directory.")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Path to the database file - using relative path from Data directory
    db_path = 'joined_data/merged_data.db'
    analyze_data(db_path) 