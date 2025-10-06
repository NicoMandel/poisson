import numpy as np
import cv2
import glob
from config import CHECKERBOARD_DIMS, IMAGE_DIR, SQUARE_SIZE_MM

# --- Parameters ---
checkerboard_dims = CHECKERBOARD_DIMS
SQUARE_SIZE_MM = SQUARE_SIZE_MM
image_dir = IMAGE_DIR
# ------------------

# Prepare object points (now scaled to real-world mm)
objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1,2)
objp = objp * SQUARE_SIZE_MM

# Arrays to store points
objpoints = []
imgpoints = []

images = glob.glob(image_dir)
gray = None # Initialize gray to ensure it has a value

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

if not objpoints:
    print("\nError: No chessboard corners were found in any images. Calibration cannot proceed.")
    print("Please check your CHECKERBOARD_DIMS, image quality, and lighting.")
else:
    # --- Perform Calibration ---
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("\n" + "="*40)
        print("      Calibration Successful!")
        print(f"\nOverall Mean Re-projection Error: {ret:.4f} pixels")
        print("="*40 + "\n")
        print("0.0-0.5 excellent")
        print("0.5-1.0 sufficient")
        print(">1.00 bad")
        print("\n"+"="*40 + "\n")

        # --- Save ONLY the camera matrix and distortion coefficients ---
        # The pixels_per_mm ratio will be calculated in the main analysis script
        # based on the specific video being analyzed for higher accuracy.
        np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist)
        print("Calibration data (mtx, dist) saved to camera_calibration_data.npz")
    else:
        print("\nCalibration failed.")