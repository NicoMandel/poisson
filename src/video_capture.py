#video_capture.py

import cv2

# --- Parameters ---
camera_index = 1      # Default webcam is usually 0. Change if you have multiple cameras.
output_filename = 'test.mp4'
frames_per_second = 20.0 # Standard is around 20-30 fps

# --- Initialization ---
# Initialize video capture
cap = cv2.VideoCapture(camera_index)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
# The 'mp4v' FOURCC is a good choice for .mp4 files.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_filename, fourcc, frames_per_second, frame_size)

print("Recording video... Press 'q' to stop.")

# --- Recording Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Write the frame to the output file
    writer.write(frame)
    
    # Display the resulting frame
    cv2.imshow('Live Recording - Press Q to stop', frame)
    
    # Check if the 'q' key was pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print(f"Video saved to {output_filename}")
cap.release()
writer.release()
cv2.destroyAllWindows()