# capture_images.py
import cv2
import os
from config import OUTPUT_DIR, NUM_IMG_CAPTURE, CAM_INDEX

# --- Parameters ---
output_dir = OUTPUT_DIR
num_images_to_capture = NUM_IMG_CAPTURE
cam_index = CAM_INDEX
# ------------------

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(cam_index)
img_counter = 0

print("Press SPACE to capture an image, ESC to quit.")

while img_counter < num_images_to_capture:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow('Capture Images', frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27: # ESC key
        print("Escape hit, closing...")
        break
    elif k % 256 == 32: # SPACE key
        img_name = os.path.join(output_dir, f"calib_{img_counter:02d}.png")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()