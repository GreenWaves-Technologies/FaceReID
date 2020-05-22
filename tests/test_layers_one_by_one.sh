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

layer_inputs=(
    conv1.0/input.json
    features.0.0/input.json
    features.3.squeeze.0/input.json
    features.3.expand1x1.0/input.json
    features.3.expand3x3.0/input.json
    features.4.squeeze.0/input.json
    features.4.expand1x1.0/input.json
    features.4.expand3x3.0/input.json
    features.6.squeeze.0/input.json
    features.6.expand1x1.0/input.json
    features.6.expand3x3.0/input.json
    features.7.squeeze.0/input.json
    features.7.expand1x1.0/input.json
    features.7.expand3x3.0/input.json
    features.9.squeeze.0/input.json
    features.9.expand1x1.0/input.json
    features.9.expand3x3.0/input.json
    features.10.squeeze.0/input.json
    features.10.expand1x1.0/input.json
    features.10.expand3x3.0/input.json
    features.11.squeeze.0/input.json
    features.11.expand1x1.0/input.json
    features.11.expand3x3.0/input.json
    features.12.squeeze.0/input.json
    features.12.expand1x1.0/input.json
    features.12.expand3x3.0/input.json
    global_avgpool/input.json
)

layer_outputs=(
    conv1.0/output.json
    features.2/output.json
    features.3.squeeze_activation/output.json
    features.3.expand1x1_activation/output.json
    features.3.expand3x3_activation/output.json
    features.4.squeeze_activation/output.json
    features.4.maxpool1x1/output.json
    features.4.maxpool3x3/output.json
    features.6.squeeze_activation/output.json
    features.6.expand1x1_activation/output.json
    features.6.expand3x3_activation/output.json
    features.7.squeeze_activation/output.json
    features.7.maxpool1x1/output.json
    features.7.maxpool3x3/output.json
    features.9.squeeze_activation/output.json
    features.9.expand1x1_activation/output.json
    features.9.expand3x3_activation/output.json
    features.10.squeeze_activation/output.json
    features.10.expand1x1_activation/output.json
    features.10.expand3x3_activation/output.json
    features.11.squeeze_activation/output.json
    features.11.expand1x1_activation/output.json
    features.11.expand3x3_activation/output.json
    features.12.squeeze_activation/output.json
    features.12.expand1x1_activation/output.json
    features.12.expand3x3_activation/output.json
    global_avgpool/output.json
)

if [ ${#layer_inputs[@]} -ne ${#layer_outputs[@]} ]; then
    echo "Layers inputs and outputs are not consistent"
    exit 1
fi

make_options="$@"

mkdir -p single_logs
echo "Generating Model"

make -C ../ReID-Demo clean > single_logs/model_generation.log 2>&1
make -j8 -C ../ReID-Demo reid_model >> single_logs/model_generation.log 2>&1 # to generate layers blobs for GAP

echo "Input blob; Output blob; Found outliers; Max diff; Diff summary;" > single_layer_test_summary.csv
cd single_layer_test

status=0

for (( i=0; i < ${#layer_inputs[@]}; i++ )); do
    echo "Layer $i: ${layer_inputs[i]} => ${layer_outputs[i]}"
    make $make_options clean > /dev/null 2>&1
    rm -rf input.bin output.bin
    ../../scripts/json2bin.py ../activations_dump/${layer_inputs[i]} input.bin
    make $make_options TEST_LAYER_INDEX=$i -j4 tiler_models > ../single_logs/$i.stdout.log 2>&1
    make $make_options TEST_LAYER_INDEX=$i all run >> ../single_logs/$i.stdout.log 2>&1
    echo -n "${layer_inputs[i]}; ${layer_outputs[i]}; " >> ../single_layer_test_summary.csv
    ../../scripts/compareWithBin.py ../activations_dump/${layer_outputs[i]} output.bin >> ../single_layer_test_summary.csv
    if [ $? -ne 0 ]; then
        echo ";" >> ../single_layer_test_summary.csv
        echo -e "Layer $i: ${layer_inputs[i]} => ${layer_outputs[i]} \e[31mFAILED\e[0m"
        (( status++ ))
    else
        echo -e "Layer $i: ${layer_inputs[i]} => ${layer_outputs[i]} \e[32mPASSED\e[0m"
    fi
    mv delta.csv ../single_logs/$i.delta.csv
    mv output.bin ../single_logs/$i.output.bin
done

cd ..

exit $status
