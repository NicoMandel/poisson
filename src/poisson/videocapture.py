import cv2
import os
from datetime import datetime


def capture_video(path : str = None, dev_id : int = 0):
    video = cv2.VideoCapture(dev_id)
    while True:
        ret, img = video.read()

        cv2.imshow('live video', img)
        
        key =cv2.waitKey(100)
        if key ==ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
