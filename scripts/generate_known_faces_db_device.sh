#!/bin/bash

rm -rf ./known_faces
mkdir ./known_faces


scripts_dir=$(dirname $0)

if [ "$1" = "-gapoc" ]; then
    MAKEFILE_NAME="Makefile DEVICE_CONNECTION=-ftdi BOARD_NAME=gapoc_a"
elif [ "$1" = "-gapuino" ]; then
    MAKEFILE_NAME="Makefile DEVICE_CONNECTION=-jtag BOARD_NAME=gapuino"
else
    echo "Target platform is not defined, use -gapoc for Gapoc A board and -gapuino for Gapuino board"
    exit 2
fi

make -j4 reid_model

idx=0
for face in $2/*;
do
    rm -f ../tests/first_n_layers_test/input.pgm
    rm -f ../tests/first_n_layers_test/output.bin
    "$scripts_dir/face_detection.py" "$face" ../tests/first_n_layers_test/input.pgm

#     convert -quality 100 -colorspace gray -resize 128x128\! "$face" ./layers_test/first_n_layers_test/input.pgm

    cd ../tests/first_n_layers_test
    make -f $MAKEFILE_NAME clean
    make -f $MAKEFILE_NAME -j4 run CONTROL_MACRO="-DPGM_INPUT=1"
    cd -
    cp ../tests/first_n_layers_test/input.pgm ./known_faces/person_$idx.pgm
    cp ../tests/first_n_layers_test/output.bin ./known_faces/person_$idx.bin
    face_file_name=$(basename $face)
    person_name="${face_file_name%.*}"
    echo "$person_name" >> ./known_faces/index.txt
    let "idx = idx + 1"
done
