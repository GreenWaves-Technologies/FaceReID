#!/bin/bash

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
