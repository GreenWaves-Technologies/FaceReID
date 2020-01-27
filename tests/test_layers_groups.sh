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

stop_macros=(\
"-DSTOP_AFTER_ConvLayer0=1" \
"-DSTOP_AFTER_ConvLayer1=1" \
"-DSTOP_AFTER_FIRE_MODULE=0" \
"-DSTOP_AFTER_FIRE_MODULE=1" \
"-DSTOP_AFTER_FIRE_MODULE=2" \
"-DSTOP_AFTER_FIRE_MODULE=3" \
"-DSTOP_AFTER_FIRE_MODULE=4" \
"-DSTOP_AFTER_FIRE_MODULE=5" \
"-DSTOP_AFTER_FIRE_MODULE=6" \
"-DSTOP_AFTER_FIRE_MODULE=7" \
"" )

layer_outputs=(\
conv1.0/output.json \
features.2/output.json \
features.3/output.json \
features.4/output.json \
features.6/output.json \
features.7/output.json \
features.9/output.json \
features.10/output.json \
features.11/output.json \
features.12/output.json \
global_avgpool/output.json )

let "layers_count = ${#stop_macros[@]} - 1"

if [ ${#stop_macros[@]} -ne ${#layer_outputs[@]} ]; then
    echo "Layers inputs and outputs are not consistent"
    exit 1
fi

if [ "$1" = "-gapoc" ]; then
    MAKEFILE_NAME="Makefile BOARD_NAME=gapoc_a"
elif [ "$1" = "-gapuino" ]; then
    MAKEFILE_NAME="Makefile BOARD_NAME=gapuino"
else
    echo "Target platform is not defined, use -gapoc for Gapoc A board and -gapuino for Gapuino board"
    exit 2
fi

mkdir -p groups_logs
echo "Generating Model"

make -C ../ReID-Demo clean > ./groups_logs/model_generaition.log 2>&1
make -j8 -C ../ReID-Demo reid_model >> ./groups_logs/model_generaition.log 2>&1 # to generate layers blobs for GAP

echo "Stop macro; Output blob; Found outliers; Max diff; Diff summary;" > group_layer_test_summary.csv

../scripts/json2bin.py ./activations_dump/conv1.0/input.json ./first_n_layers_test/input.bin
cd first_n_layers_test

status=0

for i in $(seq 0 $layers_count)
do
    echo "Stop word $i: ${stop_macros[$i]} => ${layer_outputs[$i]}"
    make -f $MAKEFILE_NAME clean > /dev/null 2>&1
    make -f $MAKEFILE_NAME -j4 CONTROL_MACRO="${stop_macros[$i]}" tiler_models > ../groups_logs/$i.stdout.log 2>&1
    make -f $MAKEFILE_NAME -j4 CONTROL_MACRO="${stop_macros[$i]}" run >> ../groups_logs/$i.stdout.log 2>&1
    echo -n "${stop_macros[$i]}; ${layer_outputs[$i]}; " >> ../group_layer_test_summary.csv
    ../../scripts/compareWithBin.py ../activations_dump/${layer_outputs[$i]} ./output.bin >> ../group_layer_test_summary.csv
    if [ $? -ne 0 ]; then
        echo ";" >> ../group_layer_test_summary.csv
        echo -e "Layer $i: ${stop_macros[$i]} => ${layer_outputs[$i]} \e[31mFAILED\e[0m"
        let "status = $status + 1"
    else
        echo -e "Layer $i: ${stop_macros[$i]} => ${layer_outputs[$i]} \e[32mPASSED\e[0m"
    fi
    mv ./delta.csv ../groups_logs/$i.delta.csv
    mv ./output.bin ../groups_logs/$i.output.bin
done

cd ..

exit $status
