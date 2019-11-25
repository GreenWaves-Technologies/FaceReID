#!/usr/bin/python3

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
