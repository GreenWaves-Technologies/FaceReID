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

tolerance=5

make_options="$@"

stop_macros=(
    "-DSTOP_AFTER_ConvLayer0=1"
    "-DSTOP_AFTER_ConvLayer1=1"
    "-DSTOP_AFTER_FIRE_MODULE=0"
    "-DSTOP_AFTER_FIRE_MODULE=1"
    "-DSTOP_AFTER_FIRE_MODULE=2"
    "-DSTOP_AFTER_FIRE_MODULE=3"
    "-DSTOP_AFTER_FIRE_MODULE=4"
    "-DSTOP_AFTER_FIRE_MODULE=5"
    "-DSTOP_AFTER_FIRE_MODULE=6"
    "-DSTOP_AFTER_FIRE_MODULE=7"
    ""
)

layer_outputs=(
    conv1.0/output.json
    features.2/output.json
    features.3/output.json
    features.4/output.json
    features.6/output.json
    features.7/output.json
    features.9/output.json
    features.10/output.json
    features.11/output.json
    features.12/output.json
    global_avgpool/output.json
)

if [ ${#stop_macros[@]} -ne ${#layer_outputs[@]} ]; then
    echo "Layers inputs and outputs are not consistent"
    exit 1
fi

mkdir -p groups_logs

echo "Stop macro; Output blob; Found outliers; Max diff; Diff summary;" > group_layer_test_summary.csv

cd first_n_layers_test
../../scripts/json2bin.py ../activations_dump/conv1.0/input.json input.bin

status=0

for (( i=0; i < ${#stop_macros[@]}; i++ )); do
    echo "Stop word $i: ${stop_macros[i]} => ${layer_outputs[i]}"
    make $make_options clean > /dev/null 2>&1
    make $make_options CONTROL_MACRO="${stop_macros[i]}" -j4 build > ../groups_logs/$i.stdout.log 2>&1
    make $make_options CONTROL_MACRO="${stop_macros[i]}" all run >> ../groups_logs/$i.stdout.log 2>&1
    echo -n "${stop_macros[i]}; ${layer_outputs[i]}; " >> ../group_layer_test_summary.csv
    ../../scripts/compareWithBin.py ../activations_dump/${layer_outputs[i]} output.bin $tolerance >> ../group_layer_test_summary.csv
    if [ $? -ne 0 ]; then
        echo ";" >> ../group_layer_test_summary.csv
        echo -e "Layer $i: ${stop_macros[i]} => ${layer_outputs[i]} \e[31mFAILED\e[0m"
        (( status++ ))
    else
        echo -e "Layer $i: ${stop_macros[i]} => ${layer_outputs[i]} \e[32mPASSED\e[0m"
    fi
    mv delta.csv ../groups_logs/$i.delta.csv
    mv output.bin ../groups_logs/$i.output.bin
done

cd ..

exit $status
