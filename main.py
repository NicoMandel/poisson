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
import config

# --- Parameters and File Paths ---
TRACKING_ALGORITHM = config.TRACKING_ALG
USE_GAUSSIAN_BLUR = config.GAUSSIAN_BLUR
TEMPLATE_UPDATE_THRESHOLD = getattr(config, 'TEMPLATE_UPDATE_THRESHOLD', 0.85)

DATA_DIR = config.DATA_D
RESULTS_DIR = config.RESULTS_D
CALIBRATION_FILENAME = config.CAL_FILE
SCALING_VIDEO_FILENAME = config.SCALING_VIDEO_FILE
TRACKING_VIDEO_FILENAME = config.TRACKING_VIDEO_FILE

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
# 1. Load camera calibration data
with np.load(CALIBRATION_FILE) as data:
    mtx, dist = data['mtx'], data['dist']

# 2. Perform Automatic Scaling
print("--- Step 1: Automatic Scale Calculation ---")
SCALING_VIDEO_FILE = os.path.join(DATA_DIR, SCALING_VIDEO_FILENAME)
scale_cap = cv2.VideoCapture(SCALING_VIDEO_FILE)
if not scale_cap.isOpened(): exit(f"Error: Could not open scaling video file at '{SCALING_VIDEO_FILE}'")
ret, scale_frame = scale_cap.read()
if not ret: exit(f"Error: Could not read the first frame from '{SCALING_VIDEO_FILE}'.")
scale_cap.release()

img_h, img_w = scale_frame.shape[:2]
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
scale_frame_undistorted = cv2.undistort(scale_frame, mtx, dist, None, newcameramtx)
gray_undistorted = cv2.cvtColor(scale_frame_undistorted, cv2.COLOR_BGR2GRAY)

pixels_per_mm = None
ret, corners = cv2.findChessboardCorners(gray_undistorted, config.CHECKERBOARD_DIMS, None)

if ret:
    print("Checkerboard found! Calculating robust scale...")
    corners_reshaped = corners.reshape(config.CHECKERBOARD_DIMS[1], config.CHECKERBOARD_DIMS[0], 2)
    horizontal_distances = np.linalg.norm(corners_reshaped[:, :-1] - corners_reshaped[:, 1:], axis=2)
    vertical_distances = np.linalg.norm(corners_reshaped[:-1, :] - corners_reshaped[1:, :], axis=2)
    all_distances = np.concatenate((horizontal_distances.flatten(), vertical_distances.flatten()))
    avg_pixel_distance = np.mean(all_distances)

    pixels_per_mm = avg_pixel_distance / config.SQUARE_SIZE_MM
    print(f"--> Calculated robust scale: {pixels_per_mm:.4f} pixels/mm")
else:
    print("\nFATAL ERROR: Checkerboard not found in the scaling video. Exiting.")
    exit()

# 3. Load Tracking Video for point selection
print("\n--- Step 2: Load Tracking Video and Select Points ---")
TRACKING_VIDEO_FILE = os.path.join(DATA_DIR, TRACKING_VIDEO_FILENAME)
track_cap = cv2.VideoCapture(TRACKING_VIDEO_FILE)
if not track_cap.isOpened(): exit(f"Error: Could not open tracking video at '{TRACKING_VIDEO_FILE}'")
ret, first_track_frame = track_cap.read()
if not ret: exit(f"Error: Could not read first frame.")

first_frame_undistorted = cv2.undistort(first_track_frame, mtx, dist, None, newcameramtx)

win_h, win_w = 720, 1280
window_name = f'Setup: Select 2 points | Press ENTER to confirm'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, zoom_pan_mouse_callback, {'img_shape': (img_h, img_w), 'win_w': win_w, 'win_h': win_h})

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


# --- Initialize Tracking States ---
tracked_points_px = np.array(points, dtype=np.float32)

# Specific initializations for different algorithms
ref_gray = cv2.cvtColor(first_frame_undistorted, cv2.COLOR_BGR2GRAY)
prev_gray = ref_gray.copy() # Required for LK
prev_points_lk = tracked_points_px.reshape(-1, 1, 2) # Required for LK
current_templates = tracked_points_px.copy() # Required for dynamic NCC/ECC updates

initial_pixel_distance = np.linalg.norm(tracked_points_px[0] - tracked_points_px[1])
initial_length_mm = initial_pixel_distance / pixels_per_mm
results, frame_idx = [], 0

track_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(track_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\n--- Step 3: Starting {TRACKING_ALGORITHM} Tracking --- \nTracking {total_frames} frames... (Press 'q' to stop early)")

# --- Main Tracking Loop ---
try:
    while True:
        if frame_idx >= total_frames: break
        ret, frame = track_cap.read()
        if not ret: break
        frame_idx += 1
        
        cur_undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        cur_gray = cv2.cvtColor(cur_undistorted, cv2.COLOR_BGR2GRAY)

        new_tracked_points = []
        update_ref = False

        # --- ALGORITHM ROUTING ---
        if TRACKING_ALGORITHM == 'NCC':
            for i in range(len(points)):
                last_known_pos = tracked_points_px[i]
                new_pos, score = tf.track_subset_ncc_roi(ref_gray, cur_gray, current_templates[i], last_known_pos, 31, 30, USE_GAUSSIAN_BLUR)
                new_tracked_points.append(new_pos)
                
                # Dynamic template updating
                if score < TEMPLATE_UPDATE_THRESHOLD:
                    current_templates[i] = new_pos
                    update_ref = True
                    
            tracked_points_px = np.array(new_tracked_points)
            if update_ref: ref_gray = cur_gray.copy()
            
        elif TRACKING_ALGORITHM == 'LK':
            new_points, status = tf.track_subset_lk(prev_gray, cur_gray, prev_points_lk, USE_GAUSSIAN_BLUR)
            tracked_points_px = new_points.reshape(-1, 2)
            
            # Frame-to-frame step forward
            prev_gray = cur_gray.copy()
            prev_points_lk = new_points.copy()
            
        elif TRACKING_ALGORITHM == 'ECC':
            for i in range(len(points)):
                new_pos, score = tf.track_subset_ecc(ref_gray, cur_gray, current_templates[i], 31, USE_GAUSSIAN_BLUR)
                new_tracked_points.append(new_pos)
                
                # Dynamic template updating for ECC
                if score < TEMPLATE_UPDATE_THRESHOLD:
                    current_templates[i] = new_pos
                    update_ref = True
                    
            tracked_points_px = np.array(new_tracked_points)
            if update_ref: ref_gray = cur_gray.copy()


        # --- Visualization & Logging ---
        p1_mm, p2_mm = tracked_points_px / pixels_per_mm
        current_length_mm = np.linalg.norm(p1_mm - p2_mm)
        length_change_mm = current_length_mm - initial_length_mm
        results.append({'frame': frame_idx, 'length_mm': current_length_mm, 'length_change_mm': length_change_mm})
        
        text = f"Length: {current_length_mm:.3f} mm (Change: {length_change_mm:+.4f} mm)"
        cv2.line(cur_undistorted, tuple(map(int, tracked_points_px[0])), tuple(map(int, tracked_points_px[1])), (0, 255, 0), 2)
        for p in tracked_points_px: cv2.circle(cur_undistorted, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
        cv2.putText(cur_undistorted, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(f"Tracking using {TRACKING_ALGORITHM}...", cur_undistorted)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    track_cap.release()
    cv2.destroyAllWindows()
    if results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df = pd.DataFrame(results)
        output_path = os.path.join(RESULTS_DIR, f"results_{TRACKING_ALGORITHM}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        df.to_csv(output_path, index=False)
        print(f"\nTracking stopped. Saved {len(results)} frames to '{output_path}'")