import sys
import os
import time
import uuid
import math
import datetime

import numpy as np
import cv2 # (OpenCV) computer vision functions
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import matplotlib.pyplot as plt

IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), 'data/images') # images path
print(IMAGES_FOLDER)
DATA = 'val'

classes = {
    0:'fist (poing)',
    1:'five (five fingers up)',
    2:'palm (palm up)',
    3:'thumb (thumb up)',
    4:'index (index finger up)',
    5:'italy (you know what it is)',
}


def mask_array(array, imask):
    """
    The mask_array function takes an array and a mask as its arguments.
    It returns a new array whose elements
    are those of the original array wherever the corresponding mask
    element is True, and 0 everywhere else.
    """
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output


def setup_tracker(ttype):
    """
    The setup_tracker function creates a tracker object based on
    the type of tracker specified by the user.
    The function takes in one argument, ttype, which is an integer
    that corresponds to a specific type of tracker.
    """

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'GOTURN']
    tracker_type = tracker_types[ttype]

    if tracker_type not in tracker_types:
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in tracker_types:
            print(t)
        sys.exit()

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

    return tracker


video = cv2.VideoCapture(0) # 0 is the id of the camera
if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()
if not ok:
    print("Cannot read video")
    sys.exit()



bg = frame.copy()
kernel = np.ones((3,3), np.uint8)

bounds_initial = (170, 170, 170, 170)
bbox = bounds_initial

tracking = -1

positions = {
    'hand_pose': (15, 40),
    'fps': (15, 20)
}

img_count = 0

print("Please choose the type of data you want to capture:")
for i in range(len(classes)):
    print("N - {}: {}".format(i, classes[i]))
choice = input("Choice : ")



#capture, process and display loop
while True:
    ok, frame = video.read()
    display = frame.copy()
    if not ok:
        break

    timer = cv2.getTickCount()

    #processing -> finding the absolute difference between the background and the current frame
    diff = cv2.absdiff(bg, frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #thresholding
    th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    #Opening, closing and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    img_dilation = cv2.dilate(closing, kernel, iterations=2)

    #get mask indexes
    imask = img_dilation > 0

    #get the masked image
    foreground = mask_array(frame, imask)
    foreground_display = foreground.copy()

    #Tracking
    if tracking != -1:
        print("Tracking")
        tracking, bbox = tracker.update(foreground)
        tracking = int(tracking)

    hand_crop = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    # Display the resulting frame
    p1 = (int(bbox[0]), int(bbox[1])) # top left
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) # x, y, w, h
    cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
    cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

    #Frames per second
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(foreground_display, "FPS: {:.2f}".format(fps), positions['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)
    cv2.putText(display, "FPS: {:.2f}".format(fps), positions['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)

    #display results

    cv2.imshow("Original", frame)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Dilation", img_dilation)
    cv2.imshow("Foreground", foreground)
    try:
        cv2.imshow("Hand", hand_crop)
    except:
        pass

    cv2.imshow("foreground_display", foreground_display)

    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video

    if k == 27: # ESC key for quit
        print('Stopped tracking')
        break
    elif k == 114 or k == 112:  # r or p key for background update
        print("Background updated")
        bg = frame.copy()
        bbox = bounds_initial
        tracking = -1
    elif k == 116: # t key for start tracking
        print("Start tracking")
        tracker = setup_tracker(1)
        tracking = tracker.init(frame, bbox)
    elif  k == 115: # s key for save
        img_count += 1
        fname = os.path.join(IMAGES_FOLDER, CURR_POS, "{}.jpg".format(img_count))
        cv2.imwrite(fname, hand_crop)
        print('Saved image to {}'.format(fname))
    elif k != 255:
        print(k)

cv2.destroyAllWindows()
video.release()