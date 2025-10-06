#config.py 

#calibration_setup
OUTPUT_DIR = 'calibration_images'
NUM_IMG_CAPTURE = 20
CAM_INDEX = 1

#calibration_calculation
CHECKERBOARD_DIMS = (14,14)
IMAGE_DIR='calibration_images/*.png'
SQUARE_SIZE_MM = 10

#tracking functionality

#User tracking algorithms 'NCC', 'LK'
TRACKING_ALG = 'NCC' 
GAUSSIAN_BLUR = True

#directory and files
DATA_D = 'data'
RESULTS_D = 'results'
CAL_FILE = 'camera_calibration_data.npz'


SCALING_VIDEO_FILE = 'scale.mp4'
TRACKING_VIDEO_FILE = 'test.mp4'