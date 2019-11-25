#!/usr/bin/python3
import sys
import os
import numpy
import cv2

# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
else:
    sys.exit(-1)

source = cv2.imread(s)
# person_name = os.path.splitext(s)[0]


scripts_dir = os.path.dirname(sys.argv[0])

net = cv2.dnn.readNetFromCaffe(os.path.join(scripts_dir, "deploy.prototxt"),
                               os.path.join(scripts_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel"))

frame = cv2.imread(s)
frame_height = frame.shape[0]
frame_width = frame.shape[1]

scale = 324.0/frame_width
print("Scale: ", scale)

new_size = (324, int(frame_height*scale))
frame = cv2.resize(frame, new_size)

frame_height = frame.shape[0]
frame_width = frame.shape[1]

camera_roi_top = int((frame_height/2)-122)
camera_roi_bottom = int((frame_height/2)+122)
frame = frame[camera_roi_top:camera_roi_bottom, 0:324]

frame_height = frame.shape[0]
frame_width = frame.shape[1]

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite("camera.pgm", gray)

# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, False, False)

# Run a model
net.setInput(blob)
detections = net.forward()

max_confidence = 0
max_index = -1
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > max_confidence:
        max_confidence = confidence
        max_index = i;

confidence = detections[0, 0, i, 2]
if max_confidence > conf_threshold:
    x_left_bottom = int(detections[0, 0, max_index, 3] * frame_width)
    y_left_bottom = int(detections[0, 0, max_index, 4] * frame_height)
    x_right_top = int(detections[0, 0, max_index, 5] * frame_width)
    y_right_top = int(detections[0, 0, max_index, 6] * frame_height)

    print("Face roi: (%d,%d) - (%d,%d)" % (x_left_bottom, y_left_bottom, x_right_top, y_right_top))
    print("Roi size: %dx%d"% (y_right_top-y_left_bottom, x_right_top-x_left_bottom))
