#!/usr/bin/python3

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

blob = numpy.array(data)

with open(sys.argv[2], "wb") as bin_output:
    for item in blob.flatten():
        short_val = int(item)
        b = struct.pack('<h', short_val)
        bin_output.write(b)
