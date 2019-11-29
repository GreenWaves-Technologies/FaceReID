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
    sys.exit(-1)

with open(sys.argv[1], "rt") as js_input:
    data = json.load(js_input)

gold = numpy.array(data)

gold = gold.flatten()

bin_output = open(sys.argv[2], "rb")
log = open("delta.csv", "wt")
log.write("Gold; Device, Diff;\n")

outlier_count = 0
diff_sum = 0
max_diff = 0
for g in gold:
    device_val = bin_output.read(2)
    short_val = struct.unpack("<h", device_val)[0]
    diff = abs(g - short_val)
    log.write("%d; %d; %d;\n" % (g, short_val, diff))
    diff_sum = diff_sum + diff
    if(diff > max_diff):
        max_diff = diff
    if(diff > 1):
        outlier_count = outlier_count + 1

bin_output.close()
log.close()

#print("Found outliers; Max diff; Diff summary")
print("%d; %d; %d;" % (outlier_count, max_diff, diff_sum))
if (outlier_count > 0):
    sys.exit(1)
