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
import json
import struct
import numpy

if(len(sys.argv) <= 2):
    print("Inpunt and output files are not set in command line\n")
    os.exit(-1)

with open(sys.argv[1], "rt") as js_input:
    data = json.load(js_input)

weights = numpy.array(data['weight'])

tf_weigths = weights

bias = numpy.array(data['bias'])

print("Norm: %d" % data['norm'][0])

written = 0
with open(sys.argv[2] + ".bias.bin", "wb") as bin_output:
    for bias_item in bias.flatten():
        short_val = int(bias_item)
        b = struct.pack('<h', short_val)
        bin_output.write(b)
        written = written + 2
    tail = 4 * ((written + 3) // 4) - written
    for i in range(0, tail):
        b = struct.pack('<b', 0)
        bin_output.write(b)

written = 0
with open(sys.argv[2] + ".weights.bin", "wb") as bin_output:
    for weights_item in tf_weigths.flatten():
        short_val = int(weights_item)
        b = struct.pack('<h', short_val)
        bin_output.write(b)
        written = written + 2
    tail = 4 * ((written + 3) // 4) - written
    for i in range(0, tail):
        b = struct.pack('<b', 0)
        bin_output.write(b)
