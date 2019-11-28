#!/bin/bash

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

scripts_dir=$(dirname $0)

rm -rf ./known_faces
mkdir ./known_faces

rm -rf ../tests/first_n_layers_test-host-bin
mkdir -p ../tests/first_n_layers_test-host-bin/f1/build
cd ../tests/first_n_layers_test-host-bin/f1/build
cmake -DCONTROL_MACRO="-DPGM_INPUT=1" ../../../first_n_layers_test
make -j4
cd -
cp ./quantized_model/*.bin ../tests/first_n_layers_test-host-bin/f1/build/bin/

idx=0
for face in $1/*;
do
    rm -f ../tests/first_n_layers_test-host-bin/input.pgm
    rm -f ../tests/first_n_layers_test-host-bin/output.bin
    "$scripts_dir/face_detection.py" "$face" ../tests/first_n_layers_test-host-bin/input.pgm

    cd ../tests/first_n_layers_test-host-bin/f1/build/bin/
    ./gap8-layers-test
    cd -

    cp ../tests/first_n_layers_test-host-bin/input.pgm ./known_faces/person_$idx.pgm
    cp ../tests/first_n_layers_test-host-bin/output.bin ./known_faces/person_$idx.bin
    face_file_name=$(basename $face)
    person_name="${face_file_name%.*}"
    echo "$person_name" >> ./known_faces/index.txt
    let "idx = idx + 1"
done
