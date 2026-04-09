import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- Configuration & File Paths ---
# 1. Get the absolute path of the directory this script is currently in ('src')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the main project folder, then into 'results'
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

INPUT_CSV = 'combined_shear_results.csv'
OUTPUT_CSV = 'cropped_shear_results.csv'

# Create the full paths
INPUT_PATH = os.path.join(RESULTS_DIR, INPUT_CSV)
OUTPUT_PATH = os.path.join(RESULTS_DIR, OUTPUT_CSV)

def main():
    print(f"Loading data from: {INPUT_PATH}")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"\nError: Could not find {INPUT_CSV}.")
        print(f"Make sure you have run the mapping script and the file exists in your 'results' folder.")
        return

    # Extract arrays for plotting
    machine_disp = df['Machine_Displacement_mm'].values
    optical_disp = df['Optical_Shear_Displacement_mm'].values
    force = df['Force_N'].values
    strain = df['Shear_Strain'].values
    stress = df['Shear_Stress_MPa'].values
    
    total_points = len(df)

    # --- Plot Setup ---
    # We make the bottom margin larger to leave room for the slider and button
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)

    # Plot 1: Machine Data
    ax1.plot(machine_disp, force, color='green', linewidth=1.5, alpha=0.6)
    dot1, = ax1.plot(machine_disp[0], force[0], 'ro', markersize=8) # Red tracking dot
    ax1.set_title('Machine Data: Force vs. Machine Deflection')
    ax1.set_xlabel('Machine Displacement (mm)')
    ax1.set_ylabel('Force (N)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Optical Data
    ax2.plot(optical_disp, force, color='blue', linewidth=1.5, alpha=0.6)
    dot2, = ax2.plot(optical_disp[0], force[0], 'ro', markersize=8)
    ax2.set_title('Optical Data: Force vs. True Deflection')
    ax2.set_xlabel('Optical Shear Displacement (mm)')
    ax2.set_ylabel('Force (N)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Stress vs Strain
    ax3.plot(strain, stress, color='red', linewidth=1.5, alpha=0.6)
    dot3, = ax3.plot(strain[0], stress[0], 'ko', markersize=8) # Black tracking dot for contrast
    ax3.set_title('Material Property: Shear Stress vs. Strain')
    ax3.set_xlabel('Shear Strain (-)')
    ax3.set_ylabel('Shear Stress (MPa)')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # --- Interactive Slider Setup ---
    # Define axes for the slider [left, bottom, width, height]
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    
    # Create the slider mapped to the dataframe index
    index_slider = Slider(
        ax=ax_slider,
        label='Cutoff Point',
        valmin=0,
        valmax=total_points - 1,
        valinit=total_points - 1, # Start at the end
        valstep=1,
        color='gray'
    )

    # Function to update the dots when the slider moves
    def update(val):
        idx = int(index_slider.val)
        
        # Update the position of the tracking dots
        dot1.set_data([machine_disp[idx]], [force[idx]])
        dot2.set_data([optical_disp[idx]], [force[idx]])
        dot3.set_data([strain[idx]], [stress[idx]])
        
        fig.canvas.draw_idle()

    index_slider.on_changed(update)

    # --- Save Button Setup ---
    ax_button = plt.axes([0.85, 0.08, 0.1, 0.06])
    save_button = Button(ax_button, 'Save & Crop', color='lightgreen', hovercolor='palegreen')

    # Function to slice the dataframe and save it when the button is clicked
    def save_and_close(event):
        cutoff_idx = int(index_slider.val)
        
        # Slice the dataframe from start up to the selected index
        cropped_df = df.iloc[:cutoff_idx + 1]
        
        # Save to new CSV
        # Ensure the results directory exists (just in case)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        cropped_df.to_csv(OUTPUT_PATH, index=False)
        
        print(f"\nSuccess! Cropped {total_points - (cutoff_idx + 1)} quirky data points.")
        print(f"Saved clean dataset with {len(cropped_df)} points to: {OUTPUT_PATH}")
        
        plt.close() # Close the window automatically

    save_button.on_clicked(save_and_close)

    # Show the plot and start the interactive GUI
    plt.show()

if __name__ == "__main__":
    main()