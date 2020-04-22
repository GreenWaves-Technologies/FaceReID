#!/bin/bash

# Copyright 2019-2020 GreenWaves Technologies, SAS
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

rm -rf ./known_faces
mkdir ./known_faces

scripts_dir=$(dirname $0)
MAKEFILE_NAME="Makefile"

cd "$scripts_dir/../ReID-Demo"
make -j4 reid_model
cd -

idx=0
for face in $2/*;
do
    rm -f $scripts_dir/../tests/first_n_layers_test/input.pgm
    rm -f $scripts_dir/../tests/first_n_layers_test/output.bin
    "$scripts_dir/face_detection.py" "$face" $scripts_dir/../tests/first_n_layers_test/input.pgm

#     convert -quality 100 -colorspace gray -resize 128x128\! "$face" ./layers_test/first_n_layers_test/input.pgm

    cd $scripts_dir/../tests/first_n_layers_test
    make -f $MAKEFILE_NAME clean
    make -f $MAKEFILE_NAME -j4 run CONTROL_MACRO="-DPGM_INPUT=1"
    cd -
    cp $scripts_dir/../tests/first_n_layers_test/input.pgm ./known_faces/person_$idx.pgm
    cp $scripts_dir/../tests/first_n_layers_test/output.bin ./known_faces/person_$idx.bin
    face_file_name=$(basename $face)
    person_name="${face_file_name%.*}"
    echo "$person_name" >> ./known_faces/index.txt
    let "idx = idx + 1"
done
