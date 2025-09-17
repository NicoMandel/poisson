# src/analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Parameters ---
# --- CHANGE THIS to the name of the CSV file you want to analyze ---
INPUT_CSV_FILENAME = "results_NCC_20250917-180113.csv" 
# --------------------------------------------------------------------

# --- File Paths ---
# Get the directory of the current script (src)
script_dir = os.path.dirname(__file__)
# Get the parent directory (the project root)
project_root = os.path.dirname(script_dir)
# Define the results directory relative to the project root
RESULTS_DIR = os.path.join(project_root, 'results')

# Build the full paths for input and output files
input_path = os.path.join(RESULTS_DIR, INPUT_CSV_FILENAME)

# --- Main Script ---

# 1. Load the data using pandas
try:
    df = pd.read_csv(input_path)
    print(f"Successfully loaded '{input_path}'")
except FileNotFoundError:
    print(f"Error: The file '{input_path}' was not found.")
    print("Please make sure the filename is correct and the file is in the 'results' directory.")
    exit()

# 2. Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# 3. Check data type and create the appropriate plot
if 'dx_mm' in df.columns and 'dy_mm' in df.columns:
    # This is a point displacement file
    print("Detected point displacement data. Plotting dx and dy vs. frame number...")
    ax.plot(df['frame'], df['dx_mm'], label='X Displacement (dx)', marker='o', markersize=3, linestyle='-')
    ax.plot(df['frame'], df['dy_mm'], label='Y Displacement (dy)', marker='x', markersize=3, linestyle='--')
    ax.set_title('Point Displacement Over Time', fontsize=16)
    ax.set_ylabel('Displacement (mm)', fontsize=12)
    plot_type = 'displacement'

elif 'length_change_mm' in df.columns:
    # This is a line length change file
    print("Detected line length change data. Plotting length change vs. frame number...")
    ax.plot(df['frame'], df['length_change_mm'], label='Length Change', color='purple', marker='.', linestyle='-')
    ax.set_title('Line Length Change Over Time', fontsize=16)
    ax.set_ylabel('Change in Length (mm)', fontsize=12)
    plot_type = 'length_change'
    
else:
    print("Error: Could not determine data type from CSV columns.")
    exit()

# 4. Finalize and save the plot
ax.set_xlabel('Frame Number', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(left=0) # Start x-axis at 0
fig.tight_layout()

# Save the plot to the results folder
output_plot_filename = f"plot_{plot_type}_{os.path.splitext(INPUT_CSV_FILENAME)[0]}.png"
output_plot_path = os.path.join(RESULTS_DIR, output_plot_filename)
plt.savefig(output_plot_path, dpi=300)

print(f"Plot saved successfully to '{output_plot_path}'")

# Optionally, show the plot on screen
plt.show()