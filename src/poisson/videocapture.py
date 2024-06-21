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

def detect_circles(cv_img : np.ndarray) -> np.ndarray:
    """
        https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    dp = 1.4
    mindist = 10
    circs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, mindist , param1 = 50, param2 = 30, minRadius=5, maxRadius=15)
    return circs

def draw_circles(cv_img : np.ndarray, circs) -> np.ndarray:
    if circs is not None:
        for pt in circs[0, :]:
            ctr = (int(np.floor(pt[0])), int(np.floor(pt[1])))
            rad = int(pt[2])
            # Draw center
            cv2.circle(cv_img, ctr, 1, (255, 0, 0), 3)
            # draw outline
            cv2.circle(cv_img, ctr, rad, (255, 0, 255), 3)
        
    return cv_img
