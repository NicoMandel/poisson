# run_calibration.py
import numpy as np
import cv2
import glob
from config import CHECKERBOARD_DIMS, IMAGE_DIR, SQUARE_SIZE_MM

# --- Parameters ---
checkerboard_dims = CHECKERBOARD_DIMS # (corners_wide, corners_high)
SQUARE_SIZE_MM = SQUARE_SIZE_MM # <--- SET THIS to the real-world size of your checker squares in mm
image_dir = IMAGE_DIR
# ------------------

# Prepare object points
objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1,2)
objp = objp * SQUARE_SIZE_MM # Scale object points to real-world size

# Arrays to store points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

images = glob.glob(image_dir)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)
    if ret:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        print(f"Corners found in {fname}")
    else:
        print(f"Corners NOT found in {fname}")

# --- Perform Calibration ---
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("\n" + "="*40)
    print("      Calibration Successful!")
    print(f"\nOverall Mean Re-projection Error: {ret:.4f} pixels")
    print("="*40 + "\n")

    # --- NEW: Calculate Pixels-per-Millimeter Ratio ---
    print("Calculating pixels-per-mm ratio...")
    # Use the first image for this calculation
    first_image_path = images[0]
    img = cv2.imread(first_image_path)
    
    # Undistort the image to get accurate measurements
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # Re-find corners on the undistorted image
    gray_undistorted = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    ret, corners_undistorted = cv2.findChessboardCorners(gray_undistorted, checkerboard_dims, None)
    
    if ret:
        # Get the first two adjacent corners
        corner1 = corners_undistorted[0][0] # Top-left corner
        corner2 = corners_undistorted[1][0] # Corner to its right
        
        # Calculate the Euclidean distance in pixels
        pixel_distance = np.linalg.norm(corner1 - corner2)
        
        # Calculate the ratio
        pixels_per_mm = pixel_distance / SQUARE_SIZE_MM
        
        print(f"  - Pixel distance between corners: {pixel_distance:.2f} px")
        print(f"  - Known square size: {SQUARE_SIZE_MM} mm")
        print(f"  - Calculated Ratio: {pixels_per_mm:.2f} pixels/mm\n")
        
        # --- Save ALL calibration data ---
        np.savez('camera_calibration_data.npz', 
                 mtx=mtx, 
                 dist=dist, 
                 pixels_per_mm=pixels_per_mm)
        print("Calibration data (including pixels_per_mm) saved to camera_calibration_data.npz")
    else:
        print("Could not find corners on the first undistorted image to calculate scale.")
        # Save without the ratio if calculation fails
        np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist)
        print("Calibration data (WITHOUT pixels_per_mm) saved.")