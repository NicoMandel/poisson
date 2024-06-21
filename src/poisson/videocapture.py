import numpy as np
import cv2


def detect_corners(cv_img : np.ndarray) -> np.ndarray:
    """
        https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # finding corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2,3, 0.04)
    dst = cv2.dilate(dst, None)

    return dst

def detect_sift(cv_img : np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des