import os
import sys
import matplotlib.pyplot as plt
from hexoskin_wav_loader import HexoskinWavLoader

def process_wav_file(file_path):
    """
    Example function that demonstrates how to use HexoskinWavLoader
    to process a Hexoskin WAV file.
    
    Args:
        file_path (str): Path to the Hexoskin WAV file
    """
    print(f"Processing file: {file_path}")
    
    # Create a new loader instance
    loader = HexoskinWavLoader()
    
    # Load the WAV file
    if not loader.load_wav_file(file_path):
        print("Failed to load WAV file. Exiting.")
        return
    
    # Get metadata
    metadata = loader.get_metadata()
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Get the data
    data = loader.get_data()
    print(f"\nLoaded {len(data)} data points")
    print("First 5 data points:")
    print(data.head())
    
    # Basic statistics
    print("\nBasic statistics:")
    print(data['value'].describe())
    
    # Apply a lowpass filter (e.g., remove frequencies above 10 Hz)
    print("\nApplying lowpass filter (10 Hz)...")
    loader.filter_data(lowcut=None, highcut=10.0)
    
    # Resample to 100 Hz
    print("\nResampling to 100 Hz...")
    loader.resample_data(100)
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    
    # Original plot title based on the file name
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    plt.title(f"Hexoskin Data: {file_name}")
    
    plt.plot(loader.data['timestamp'], loader.data['value'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Save the plot
    plot_path = f"{os.path.splitext(file_path)[0]}_plot.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Save to CSV
    csv_path = f"{os.path.splitext(file_path)[0]}_processed.csv"
    loader.save_to_csv(csv_path)
    print(f"Processed data saved to: {csv_path}")
    
    # Show the plot
    plt.show()


def main():
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python hexoskin_wav_example.py <path_to_wav_file>")
        print("Example: python hexoskin_wav_example.py ECG_I.wav")
        return
    
    # Get the file path from command line arguments
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Process the WAV file
    process_wav_file(file_path)


if __name__ == "__main__":
    main() 