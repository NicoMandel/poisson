import os.path
from argparse import ArgumentParser
import cv2

from poisson.videocapture import detect_corners, detect_sift

def parse_args(basedir : str):
    parser = ArgumentParser()
    # Required arguments
    videodir = os.path.join(basedir, 'data', 'videos') 
    videof = os.path.join(videodir, 'BF6A0742.MP4')
    parser.add_argument(
        "--input",
        help="""Input file. Defaults to <BF6A0742.MP4> in the ../data/videos folder""",
        type=str,
        default=videof
    )
    # optional arguments
    parser.add_argument(
        "-o",
        "--output",
        help="""Output location. Defaults to none.""",
        default=None,
    )
    return parser.parse_args()

if __name__=="__main__":
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args = parse_args(basedir)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError("File {} could not be opened".format(args.input))
    
    ret, frame = cap.read()
    while(cap.isOpened()):
        ret, frame = cap.read()
        dst = detect_corners(frame)
        # thresholding and marking green
        # frame[dst>0.01*dst.max()] = [0, 255, 0]
        kp, _ = detect_sift(frame)
        img = cv2.drawKeypoints(frame, kp, frame)
        cv2.imshow('frame', frame)
        # break condition
        if cv2.waitKey(25) & 0xFF == ord('q') or ret==False:
            cap.release()
            cv2.destroyAllWindows()
            break


    