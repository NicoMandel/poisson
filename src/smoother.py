import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- Configuration & File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

INPUT_CSV = 'cropped_shear_results.csv'
OUTPUT_CSV = 'smoothed_shear_results.csv'

INPUT_PATH = os.path.join(RESULTS_DIR, INPUT_CSV)
OUTPUT_PATH = os.path.join(RESULTS_DIR, OUTPUT_CSV)

# --- Smoothing Parameters ---
# WINDOW_LENGTH must be an odd number. Larger = smoother, but too large over-smooths.
# POLY_ORDER is the polynomial fit. 3 is standard for stress/strain data.
WINDOW_LENGTH = 51 
POLY_ORDER = 3

def main():
    print(f"Loading cropped data from: {INPUT_PATH}")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"\nError: Could not find {INPUT_CSV}.")
        print("Make sure you ran the interactive cropper and saved the file.")
        return

    # Extract original arrays
    machine_disp = df['Machine_Displacement_mm'].values
    force = df['Force_N'].values
    optical_disp = df['Optical_Shear_Displacement_mm'].values
    strain = df['Shear_Strain'].values
    stress = df['Shear_Stress_MPa'].values

    print(f"Applying Savitzky-Golay filter (Window: {WINDOW_LENGTH}, Poly: {POLY_ORDER})...")
    
    # Ensure window length isn't larger than the dataset itself
    window = min(WINDOW_LENGTH, len(df))
    if window % 2 == 0: 
        window -= 1 # Make sure it stays odd
    
    # Apply smoothing
    # We smooth the optical displacement, strain, and stress. 
    # Force and Machine Displacement are usually smooth enough from the machine hardware,
    # but we can smooth force slightly to remove machine vibration noise.
    smoothed_optical_disp = savgol_filter(optical_disp, window_length=window, polyorder=POLY_ORDER)
    smoothed_force = savgol_filter(force, window_length=window, polyorder=POLY_ORDER)
    smoothed_strain = savgol_filter(strain, window_length=window, polyorder=POLY_ORDER)
    smoothed_stress = savgol_filter(stress, window_length=window, polyorder=POLY_ORDER)

    # Save to a new DataFrame
    df_smoothed = pd.DataFrame({
        'Machine_Displacement_mm': machine_disp, # Kept original
        'Optical_Shear_Displacement_mm': smoothed_optical_disp,
        'Force_N': smoothed_force,
        'Shear_Stress_MPa': smoothed_stress,
        'Shear_Strain': smoothed_strain
    })
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_smoothed.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Saved smoothed dataset to: {OUTPUT_PATH}")

    # --- Plotting Before & After ---
    print("Generating comparison plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Machine Data
    ax1.plot(machine_disp, force, color='lightgray', label='Original', linewidth=2)
    ax1.plot(machine_disp, smoothed_force, color='green', label='Smoothed', linewidth=1.5)
    ax1.set_title('Machine Data: Force vs. Machine Deflection')
    ax1.set_xlabel('Machine Displacement (mm)')
    ax1.set_ylabel('Force (N)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Optical Data
    ax2.plot(optical_disp, force, color='lightgray', label='Original', linewidth=2)
    ax2.plot(smoothed_optical_disp, smoothed_force, color='blue', label='Smoothed', linewidth=1.5)
    ax2.set_title('Optical Data: Force vs. True Deflection')
    ax2.set_xlabel('Optical Shear Displacement (mm)')
    ax2.set_ylabel('Force (N)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Stress vs Strain
    ax3.plot(strain, stress, color='lightgray', label='Original', linewidth=2)
    ax3.plot(smoothed_strain, smoothed_stress, color='red', label='Smoothed', linewidth=1.5)
    ax3.set_title('Material Property: Shear Stress vs. Strain')
    ax3.set_xlabel('Shear Strain (-)')
    ax3.set_ylabel('Shear Stress (MPa)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()