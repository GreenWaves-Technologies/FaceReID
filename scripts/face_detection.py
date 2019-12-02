#!/usr/bin/python3

# Copyright 2019 GreenWaves Technologies, SAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
if len(sys.argv) > 2:
    s = sys.argv[1]
    target = sys.argv[2]
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

# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, False, False)

# Run a model
net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
        y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
        x_right_top = int(detections[0, 0, i, 5] * frame_width)
        y_right_top = int(detections[0, 0, i, 6] * frame_height)

        print(x_left_bottom, y_left_bottom, (x_right_top-x_left_bottom), (y_right_top-y_left_bottom))

        face_roi = frame[y_left_bottom:y_right_top, x_left_bottom:x_right_top]
        resized = cv2.resize(face_roi, (128,128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(target, gray)
