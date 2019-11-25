#!/bin/bash

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
