import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Configuration & File Paths ---
RESULTS_DIR = 'results'
OPTICAL_CSV_FILENAME = 'tracking_results.csv'  # Your camera tracking data
MACHINE_CSV_FILENAME = 'FCC_0_1.csv'           # Your newly provided machine data

OPTICAL_CSV_PATH = os.path.join(RESULTS_DIR, OPTICAL_CSV_FILENAME)
MACHINE_CSV_PATH = os.path.join(RESULTS_DIR, MACHINE_CSV_FILENAME)

# Specimen Parameters
AREA_MM2 = 40.0 * 40.0  # 40x40mm = 1600 mm^2

def main():
    print("Loading data...")
    
    # 1. Load Optical CSV Data (Camera Tracking)
    try:
        df_optical = pd.read_csv(OPTICAL_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {OPTICAL_CSV_PATH}.")
        return

    # Shear displacement is the change in length
    optical_displacement_mm = df_optical['length_change_mm'].values
    
    # We use the very first length as the initial gap width (L0) to calculate strain
    initial_length_mm = df_optical['length_mm'].iloc[0] 

    # 2. Load Machine CSV Data (Tensile Tester)
    print(f"Loading machine data from {MACHINE_CSV_FILENAME}...")
    try:
        # Based on your provided data: time_sec, x_mm, F_N
        df_machine = pd.read_csv(MACHINE_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {MACHINE_CSV_PATH}.")
        return

    # Drop any trailing NaN rows just in case
    df_machine = df_machine.dropna()
    
    # Map the columns to our variables
    force_N = df_machine['F_N'].values
    machine_displacement_mm = df_machine['x_mm'].values

    # 3. Interpolate/Extrapolate Optical data to match Machine data length
    print("Synchronizing and interpolating datasets...")
    n_optical = len(optical_displacement_mm)
    n_machine = len(force_N)

    # Create normalized arrays (0.0 to 1.0) representing the start and end of the test
    t_optical = np.linspace(0, 1, n_optical)
    t_machine = np.linspace(0, 1, n_machine)

    # Create interpolation function based on the camera data
    interpolator = interp1d(t_optical, optical_displacement_mm, kind='linear', fill_value="extrapolate")
    
    # Generate the matched optical shear displacement array
    matched_optical_displacement = interpolator(t_machine)

    # 4. Calculate Mechanics
    print("Calculating shear stress and strain...")
    
    # Shear Stress (Tau) = Force / Area. 
    # Force in N / Area in mm^2 = Stress in MPa.
    shear_stress_MPa = force_N / AREA_MM2
    
    # Shear Strain (Gamma) = Displacement / Initial Length
    shear_strain = matched_optical_displacement / initial_length_mm

    # Optional: Save the combined calculated data to a new CSV
    df_combined = pd.DataFrame({
        'Machine_Displacement_mm': machine_displacement_mm,
        'Optical_Shear_Displacement_mm': matched_optical_displacement,
        'Force_N': force_N,
        'Shear_Stress_MPa': shear_stress_MPa,
        'Shear_Strain': shear_strain
    })
    output_path = os.path.join(RESULTS_DIR, 'combined_shear_results.csv')
    df_combined.to_csv(output_path, index=False)
    print(f"Combined dataset saved to: {output_path}")

    # 5. Plotting
    print("Generating plots...")
    
    # Create a figure with 3 subplots side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Force vs Machine Deflection (Raw machine data)
    ax1.plot(machine_displacement_mm, force_N, color='green', linewidth=2)
    ax1.set_title('Machine Data: Force vs. Machine Deflection')
    ax1.set_xlabel('Machine Displacement (x_mm)')
    ax1.set_ylabel('Force (F_N)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Force vs Optical Deflection (Camera data)
    ax2.plot(matched_optical_displacement, force_N, color='blue', linewidth=2)
    ax2.set_title('Optical Data: Force vs. True Deflection')
    ax2.set_xlabel('Optical Shear Displacement (mm)')
    ax2.set_ylabel('Force (F_N)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Calculated Shear Stress vs Shear Strain
    ax3.plot(shear_strain, shear_stress_MPa, color='red', linewidth=2)
    ax3.set_title('Material Property: Shear Stress vs. Strain')
    ax3.set_xlabel('Shear Strain (-)')
    ax3.set_ylabel('Shear Stress (MPa)')
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()