import sys
import os

import time
import uuid
import math
import argparse
import cv2

import numpy as np
from utils import setup_tracker

parser = argparse.ArgumentParser(description='Tracker') # create parser
parser.add_argument('src', default='video')
args = parser.parse_args()

try:
    src = int(args.src)
    video = cv2.VideoCapture(src)

except ValueError:
    if args.src == 'running':
        video = cv2.VideoCapture(os.path.join('image', 'running.mp4'))
    elif args.src == 'bottle':
        video = cv2.VideoCapture(os.path.join('image', 'moving_subject.mp4'))
    else:
        raise Exception('Invalid source')

cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

success, frame = video.read()
if not success:
    print('Cannot read video')
    sys.exit()

tracker = setup_tracker(4)
bbox = cv2.selectROI('frame', frame, False)
cv2.destroyAllWindows()

tracking_success = tracker.init(frame, bbox)

while True:
    time.sleep(0.02)

    timer = cv2.getTickCount()

    success, frame = video.read()
    if not success:
        break

    tracking_success, bbox = tracker.update(frame)

    if tracking_success:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break # ESC pressed

cv2.destroyAllWindows()
video.release()