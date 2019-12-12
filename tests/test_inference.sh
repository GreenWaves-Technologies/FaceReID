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

tolerance=4

if [ "$1" = "-gapoc" ]; then
    MAKEFILE_NAME="Makefile BOARD_NAME=gapoc_a"
elif [ "$1" = "-gapuino" ]; then
    MAKEFILE_NAME="Makefile BOARD_NAME=gapuino"
else
    echo "Target platform is not defined, use -gapoc for Gapoc A board and -gapuino for Gapuino board"
    exit 2
fi

mkdir -p inference_logs
echo "Generating Model"

make -C ../ReID-Demo clean > ./inference_logs/model_generaition.log 2>&1
make -j8 -C ../ReID-Demo reid_model >> ./inference_logs/model_generaition.log 2>&1 # to generate layers blobs for GAP

../scripts/json2bin.py ./activations_dump/conv1.0/input.json ./first_n_layers_test/input.bin
cd first_n_layers_test

make -f $MAKEFILE_NAME clean > /dev/null 2>&1
make -f $MAKEFILE_NAME -j4 tiler_models > ../inference_logs/stdout.log 2>&1
make -f $MAKEFILE_NAME -j4 run >> ../inference_logs/stdout.log 2>&1
../../scripts/compareWithBin.py ../activations_dump/global_avgpool/output.json ./output.bin $tolerance > ../inference_test_summary.csv
if [ $? -ne 0 ]; then
    echo ";" >> ../inference_test_summary.csv
    echo -e "Inference test \e[31mFAILED\e[0m"
    let "status = $status + 1"
else
    echo -e "Inference test \e[32mPASSED\e[0m"
fi
mv ./delta.csv ../inference_logs/delta.csv
mv ./output.bin ../inference_logs/output.bin

cd ..

exit $status
