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

mkdir -p inference_logs

cd first_n_layers_test
../../scripts/json2bin.py ../activations_dump/conv1.0/input.json input.bin

make $make_options clean > /dev/null 2>&1
make $make_options -j4 build > ../inference_logs/stdout.log 2>&1
make $make_options all run >> ../inference_logs/stdout.log 2>&1
../../scripts/compareWithBin.py ../activations_dump/global_avgpool/output.json output.bin $tolerance > ../inference_test_summary.csv
if [ $? -ne 0 ]; then
    echo ";" >> ../inference_test_summary.csv
    echo -e "Inference test \e[31mFAILED\e[0m"
    (( status++ ))
else
    echo -e "Inference test \e[32mPASSED\e[0m"
fi
mv delta.csv ../inference_logs/delta.csv
mv output.bin ../inference_logs/output.bin

cd ..

exit $status
