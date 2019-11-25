#!/usr/bin/python3

import sys
import os
import json
import numpy

if(len(sys.argv) <= 1):
    print("Inpunt and output files are not set in command line\n")
    sys.exit(-1)

with open(sys.argv[1], "rt") as js_input:
    data = json.load(js_input)

blob = numpy.array(data)
print(blob.shape)
