# main.py
import cv2
import numpy as np
import pandas as pd
import time
import sys
import os

# --- Add 'src' directory to Python's path ---
sys.path.append('src')
import tracking_functions as tf

# --- Parameters and File Paths ---
TRACKING_ALGORITHM = 'NCC'
USE_GAUSSIAN_BLUR = True
DATA_DIR = 'data'
RESULTS_DIR = 'results'
VIDEO_FILENAME = 'example_video.mp4'
CALIBRATION_FILENAME = 'camera_calibration_data.npz'

# Build full, cross-platform paths
VIDEO_FILE = os.path.join(DATA_DIR, VIDEO_FILENAME)
CALIBRATION_FILE = os.path.join(DATA_DIR, CALIBRATION_FILENAME)

# --- State for Zoom/Pan Interface ---
zoom_state = {'level': 1.0, 'center_x': 0.5, 'center_y': 0.5, 'panning': False, 'pan_start': (0,0)}
points = [] # Global for mouse selection

def zoom_pan_mouse_callback(event, x, y, flags, param):
    """Handles mouse events for the zoom/pan interface."""
    global points, zoom_state
    # (Mouse callback logic is unchanged)
    if event == cv2.EVENT_MOUSEWHEEL:
        img_h, img_w = param['img_shape']
        img_x = int((x / param['win_w'] - 0.5) * img_w / zoom_state['level'] + zoom_state['center_x'] * img_w)
        img_y = int((y / param['win_h'] - 0.5) * img_h / zoom_state['level'] + zoom_state['center_y'] * img_h)
        if flags > 0: zoom_state['level'] *= 1.1
        else: zoom_state['level'] /= 1.1
        zoom_state['level'] = max(1.0, zoom_state['level'])
        zoom_state['center_x'] = img_x / img_w; zoom_state['center_y'] = img_y / img_h
    if event == cv2.EVENT_RBUTTONDOWN:
        zoom_state['panning'] = True; zoom_state['pan_start'] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and zoom_state['panning']:
        dx = (x - zoom_state['pan_start'][0]); dy = (y - zoom_state['pan_start'][1])
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
    # (get_zoomed_view function is unchanged)
    img_h, img_w = full_img.shape[:2]
    zoom['center_x'] = np.clip(zoom['center_x'], (0.5 / zoom['level']), 1 - (0.5 / zoom['level']))
    zoom['center_y'] = np.clip(zoom['center_y'], (0.5 / zoom['level']), 1 - (0.5 / zoom['level']))
    crop_w = int(img_w / zoom['level']); crop_h = int(img_h / zoom['level'])
    crop_x = int(zoom['center_x'] * img_w - crop_w / 2); crop_y = int(zoom['center_y'] * img_h - crop_h / 2)
    cropped = full_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    return cv2.resize(cropped, (win_w, win_h))

# --- Main Application ---
# ... (Setup and point selection logic is unchanged) ...
with np.load(CALIBRATION_FILE) as data: mtx, dist, pixels_per_mm = data['mtx'], data['dist'], data['pixels_per_mm']
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened(): print(f"Error: Could not open video file at '{VIDEO_FILE}'"); exit()
ret, first_frame = cap.read()
if not ret: print(f"Error: Could not read the first frame from '{VIDEO_FILE}'."); cap.release(); exit()
img_h, img_w = first_frame.shape[:2]
newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 1, (img_w, img_h))
first_frame_undistorted = cv2.undistort(first_frame, mtx, dist, None, newcameramtx)
win_h, win_w = 720, 1280
window_name = 'Setup: Zoom/Pan with Mouse Wheel/R-Click | Select with L-Click'
cv2.namedWindow(window_name)
callback_params = {'img_shape': (img_h, img_w), 'win_w': win_w, 'win_h': win_h}
cv2.setMouseCallback(window_name, zoom_pan_mouse_callback, callback_params)
while True:
    view = get_zoomed_view(first_frame_undistorted, win_w, win_h, zoom_state)
    for p in points:
        view_x_start = zoom_state['center_x'] * img_w - (img_w / zoom_state['level']) / 2
        view_y_start = zoom_state['center_y'] * img_h - (img_h / zoom_state['level']) / 2
        win_x = int((p[0] - view_x_start) * zoom_state['level'] * (win_w / img_w))
        win_y = int((p[1] - view_y_start) * zoom_state['level'] * (win_h / img_h))
        cv2.circle(view, (win_x, win_y), 5, (0, 0, 255), -1)
    cv2.imshow(window_name, view)
    if cv2.waitKey(1) & 0xFF == 13 and len(points) > 0: break # ENTER
cv2.destroyAllWindows()
selection_mode = "point" if len(points) == 1 else "line"
tracked_points_px = np.array(points, dtype=np.float32)
ref_gray = cv2.cvtColor(first_frame_undistorted, cv2.COLOR_BGR2GRAY)
initial_points_mm = np.array(points) / pixels_per_mm
initial_length_mm = 0
if selection_mode == "line": initial_length_mm = np.linalg.norm(initial_points_mm[0] - initial_points_mm[1])
results = []
frame_idx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Starting tracking for {total_frames} frames... (Press 'q' or Ctrl+C to stop early and save)")

# === NEW: Wrap the tracking loop in a try...finally block ===
try:
    while True:
        if frame_idx >= total_frames: print("Reached the end of the video based on frame count."); break
        ret, frame = cap.read()
        if not ret: print("End of stream signal received."); break
        frame_idx += 1
        current_frame_undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        cur_gray = cv2.cvtColor(current_frame_undistorted, cv2.COLOR_BGR2GRAY)
        if TRACKING_ALGORITHM == 'NCC':
            new_tracked_points = []
            for i, p_start in enumerate(points):
                new_pos, _ = tf.track_subset_ncc(ref_gray, cur_gray, p_start, 31, USE_GAUSSIAN_BLUR)
                new_tracked_points.append(new_pos)
            tracked_points_px = np.array(new_tracked_points)
        elif TRACKING_ALGORITHM == 'LK':
            new_points, status = tf.track_subset_lk(ref_gray, cur_gray, tracked_points_px.reshape(-1, 1, 2), USE_GAUSSIAN_BLUR)
            if status.all() == 1: tracked_points_px = new_points.reshape(-1, 2)
            ref_gray = cur_gray.copy()
        if selection_mode == "point":
            disp_mm = (np.array(tracked_points_px[0]) / pixels_per_mm) - initial_points_mm[0]
            text = f"dx: {disp_mm[0]:.3f} mm, dy: {disp_mm[1]:.3f} mm"
            results.append({'frame': frame_idx, 'dx_mm': disp_mm[0], 'dy_mm': disp_mm[1]})
        else:
            p1_mm, p2_mm = np.array(tracked_points_px) / pixels_per_mm
            length_change_mm = np.linalg.norm(p1_mm - p2_mm) - initial_length_mm
            text = f"Length Change: {length_change_mm:.4f} mm"
            results.append({'frame': frame_idx, 'length_change_mm': length_change_mm})
        for p in tracked_points_px: cv2.circle(current_frame_undistorted, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
        if selection_mode == "line": cv2.line(current_frame_undistorted, tuple(map(int, tracked_points_px[0])), tuple(map(int, tracked_points_px[1])), (0, 255, 0), 2)
        cv2.putText(current_frame_undistorted, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Tracking...", current_frame_undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nInterrupted by user.")
            break

finally:
    # --- This block will run no matter how the loop exits ---
    cap.release()
    cv2.destroyAllWindows()
    if results:
        if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
        df = pd.DataFrame(results)
        output_filename = f"results_{TRACKING_ALGORITHM}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        df.to_csv(output_path, index=False)
        print(f"\nTracking stopped. {len(results)} frames of data saved to '{output_path}'")
    else:
        print("\nNo results to save.")
# =================================================================