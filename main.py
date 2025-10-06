import cv2
import numpy as np
import pandas as pd
import time
import sys
import os

# --- Add 'src' directory to Python's path ---
sys.path.append('src')
import tracking_functions as tf
# --- Import calibration parameters from config ---
from config import CHECKERBOARD_DIMS, SQUARE_SIZE_MM, TRACKING_ALG, GAUSSIAN_BLUR, DATA_D, RESULTS_D, CAL_FILE, SCALING_VIDEO_FILE, TRACKING_VIDEO_FILE 

# --- Parameters and File Paths ---
TRACKING_ALGORITHM = TRACKING_ALG
USE_GAUSSIAN_BLUR = GAUSSIAN_BLUR


DATA_DIR = DATA_D
RESULTS_DIR = RESULTS_D
CALIBRATION_FILENAME = CAL_FILE

# 1. A short video of the checkerboard at the specimen's focal plane
SCALING_VIDEO_FILENAME = SCALING_VIDEO_FILE
# 2. The video of the actual test with the deforming specimen
TRACKING_VIDEO_FILENAME = TRACKING_VIDEO_FILE

CALIBRATION_FILE = os.path.join(DATA_DIR, CALIBRATION_FILENAME)

# --- State for Zoom/Pan Interface ---
zoom_state = {'level': 1.0, 'center_x': 0.5, 'center_y': 0.5, 'panning': False, 'pan_start': (0,0)}
points = []

def zoom_pan_mouse_callback(event, x, y, flags, param):
    global points, zoom_state
    if event == cv2.EVENT_MOUSEWHEEL:
        img_h, img_w = param['img_shape']
        img_x = int((x / param['win_w'] - 0.5) * img_w / zoom_state['level'] + zoom_state['center_x'] * img_w)
        img_y = int((y / param['win_h'] - 0.5) * img_h / zoom_state['level'] + zoom_state['center_y'] * img_h)
        if flags > 0: zoom_state['level'] *= 1.1
        else: zoom_state['level'] /= 1.1
        zoom_state['level'] = max(1.0, zoom_state['level'])
        zoom_state['center_x'] = img_x / img_w
        zoom_state['center_y'] = img_y / img_h
    if event == cv2.EVENT_RBUTTONDOWN:
        zoom_state['panning'] = True
        zoom_state['pan_start'] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and zoom_state['panning']:
        dx, dy = (x - zoom_state['pan_start'][0]), (y - zoom_state['pan_start'][1])
        img_h, img_w = param['img_shape']
        zoom_state['center_x'] -= (dx / img_w) / zoom_state['level']
        zoom_state['center_y'] -= (dy / img_h) / zoom_state['level']
        zoom_state['pan_start'] = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        zoom_state['panning'] = False
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            img_h, img_w = param['img_shape']
            img_x = int((x / param['win_w'] - 0.5) * img_w / zoom_state['level'] + zoom_state['center_x'] * img_w)
            img_y = int((y / param['win_h'] - 0.5) * img_h / zoom_state['level'] + zoom_state['center_y'] * img_h)
            points.append((img_x, img_y))
            print(f"Point {len(points)} selected at full-res coordinate: ({img_x}, {img_y})")

def get_zoomed_view(full_img, win_w, win_h, zoom):
    img_h, img_w = full_img.shape[:2]
    zoom['center_x'] = np.clip(zoom['center_x'], 0.5/zoom['level'], 1-0.5/zoom['level'])
    zoom['center_y'] = np.clip(zoom['center_y'], 0.5/zoom['level'], 1-0.5/zoom['level'])
    crop_w, crop_h = int(img_w / zoom['level']), int(img_h / zoom['level'])
    crop_x, crop_y = int(zoom['center_x']*img_w - crop_w/2), int(zoom['center_y']*img_h - crop_h/2)
    return cv2.resize(full_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w], (win_w, win_h))

# --- Main Application ---
# 1. Load camera calibration data (mtx and dist)
with np.load(CALIBRATION_FILE) as data:
    mtx, dist = data['mtx'], data['dist']

# 2. Perform Automatic Scaling using the scaling video
print("--- Step 1: Automatic Scale Calculation ---")
SCALING_VIDEO_FILE = os.path.join(DATA_DIR, SCALING_VIDEO_FILENAME)
scale_cap = cv2.VideoCapture(SCALING_VIDEO_FILE)
if not scale_cap.isOpened(): exit(f"Error: Could not open scaling video file at '{SCALING_VIDEO_FILE}'")
ret, scale_frame = scale_cap.read()
if not ret: exit(f"Error: Could not read the first frame from '{SCALING_VIDEO_FILE}'.")
scale_cap.release() # We only need the first frame

# Undistort the scaling frame
img_h, img_w = scale_frame.shape[:2]
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
scale_frame_undistorted = cv2.undistort(scale_frame, mtx, dist, None, newcameramtx)
gray_undistorted = cv2.cvtColor(scale_frame_undistorted, cv2.COLOR_BGR2GRAY)

pixels_per_mm = None
ret, corners = cv2.findChessboardCorners(gray_undistorted, CHECKERBOARD_DIMS, None)

if ret:
    print("Checkerboard found in scaling video! Calculating robust scale...")
    corners_reshaped = corners.reshape(CHECKERBOARD_DIMS[1], CHECKERBOARD_DIMS[0], 2)
    horizontal_distances = np.linalg.norm(corners_reshaped[:, :-1] - corners_reshaped[:, 1:], axis=2)
    vertical_distances = np.linalg.norm(corners_reshaped[:-1, :] - corners_reshaped[1:, :], axis=2)
    
    # ** FIX: Concatenate the two distance arrays into one before averaging **
    all_distances = np.concatenate((horizontal_distances.flatten(), vertical_distances.flatten()))
    avg_pixel_distance = np.mean(all_distances)

    pixels_per_mm = avg_pixel_distance / SQUARE_SIZE_MM
    print(f"--> Calculated robust scale: {pixels_per_mm:.4f} pixels/mm")
else:
    # ** NEW: Fail-fast logic if checkerboard is not found **
    print("\n" + "="*60)
    print("FATAL ERROR: Checkerboard not found in the scaling video.")
    print(f"Could not automatically determine the scale from '{SCALING_VIDEO_FILENAME}'.")
    print("\nTroubleshooting:")
    print("  1. Ensure the checkerboard is flat, well-lit, and in focus.")
    print("  2. Check that CHECKERBOARD_DIMS in 'config.py' is correct.")
    print("  3. Ensure the entire checkerboard is visible in the first frame.")
    print("\nExiting program.")
    print("="*60)
    exit()

# 3. Load Tracking Video for point selection
print("\n--- Step 2: Load Tracking Video and Select Points ---")
TRACKING_VIDEO_FILE = os.path.join(DATA_DIR, TRACKING_VIDEO_FILENAME)
track_cap = cv2.VideoCapture(TRACKING_VIDEO_FILE)
if not track_cap.isOpened(): exit(f"Error: Could not open tracking video at '{TRACKING_VIDEO_FILE}'")
ret, first_track_frame = track_cap.read()
if not ret: exit(f"Error: Could not read first frame from '{TRACKING_VIDEO_FILE}'.")

# Undistort the first frame of the tracking video
img_h, img_w = first_track_frame.shape[:2] # Re-get shape in case it's different
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
first_frame_undistorted = cv2.undistort(first_track_frame, mtx, dist, None, newcameramtx)

win_h, win_w = 720, 1280
prompt = 'Select 2 points to track'
window_name = f'Setup: {prompt} | Press ENTER to confirm'
cv2.namedWindow(window_name)
callback_params = {'img_shape': (img_h, img_w), 'win_w': win_w, 'win_h': win_h}
cv2.setMouseCallback(window_name, zoom_pan_mouse_callback, callback_params)

while True:
    view = get_zoomed_view(first_frame_undistorted, win_w, win_h, zoom_state)
    for p in points:
        view_x_start = zoom_state['center_x']*img_w - (img_w/zoom_state['level'])/2
        view_y_start = zoom_state['center_y']*img_h - (img_h/zoom_state['level'])/2
        win_x = int((p[0]-view_x_start) * zoom_state['level'] * (win_w/img_w))
        win_y = int((p[1]-view_y_start) * zoom_state['level'] * (win_h/img_h))
        cv2.circle(view, (win_x, win_y), 5, (0, 0, 255), -1)
    cv2.imshow(window_name, view)
    if cv2.waitKey(1) & 0xFF == 13 and len(points) == 2: break
cv2.destroyAllWindows()


# --- Initialize Tracking ---
tracked_points_px = np.array(points, dtype=np.float32)
ref_gray = cv2.cvtColor(first_frame_undistorted, cv2.COLOR_BGR2GRAY)
initial_pixel_distance = np.linalg.norm(tracked_points_px[0] - tracked_points_px[1])
initial_length_mm = initial_pixel_distance / pixels_per_mm
results, frame_idx = [], 0
track_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(track_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\n--- Step 3: Starting Tracking --- \nTracking {total_frames} frames... (Press 'q' to stop early)")

# --- Main Tracking Loop ---
try:
    while True:
        if frame_idx >= total_frames: break
        ret, frame = track_cap.read();
        if not ret: break
        frame_idx += 1
        
        cur_undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        cur_gray = cv2.cvtColor(cur_undistorted, cv2.COLOR_BGR2GRAY)

        new_tracked_points = []
        for p_start in points:
            new_pos, _ = tf.track_subset_ncc(ref_gray, cur_gray, p_start, 31, USE_GAUSSIAN_BLUR)
            new_tracked_points.append(new_pos)
        tracked_points_px = np.array(new_tracked_points)
        
        p1_mm, p2_mm = tracked_points_px / pixels_per_mm
        current_length_mm = np.linalg.norm(p1_mm - p2_mm)
        length_change_mm = current_length_mm - initial_length_mm
        results.append({'frame': frame_idx, 'length_mm': current_length_mm, 'length_change_mm': length_change_mm})
        
        text = f"Length: {current_length_mm:.3f} mm (Change: {length_change_mm:+.4f} mm)"
        cv2.line(cur_undistorted, tuple(map(int, tracked_points_px[0])), tuple(map(int, tracked_points_px[1])), (0, 255, 0), 2)
        for p in tracked_points_px: cv2.circle(cur_undistorted, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
        cv2.putText(cur_undistorted, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Tracking...", cur_undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    track_cap.release()
    cv2.destroyAllWindows()
    if results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df = pd.DataFrame(results)
        output_filename = f"results_{TRACKING_ALGORITHM}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        df.to_csv(output_path, index=False)
        print(f"\nTracking stopped. Saved {len(results)} frames to '{output_path}'")
