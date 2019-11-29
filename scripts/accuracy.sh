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

rm -rf ./accuracy
mkdir ./accuracy

idx=0
for face in $1/*;
do
    face_file_name=$(basename $face)

    echo $face
    echo $face_file_name

    rm -f ./tests/first_n_layers_test/input.pgm
    rm -f ./tests/first_n_layers_test/output.bin
    cp "$face" ./tests/first_n_layers_test/input.pgm

    cd ./tests/first_n_layers_test
    make clean
    make -j4 run CONTROL_MACRO="-DPGM_INPUT=1"
    cd -
    cp ./tests/first_n_layers_test/output.bin ./accuracy/$face_file_name.bin
done
