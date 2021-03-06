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

make_options="$@"

mkdir -p prepare_pipeline_logs

cd prepare_pipeline_test

status=0
for i in {0..2}; do
    echo "Prepare L$i test"
    make $make_options TEST_LEVEL=$i clean > /dev/null 2>&1
    make $make_options TEST_LEVEL=$i -j4 build > ../prepare_pipeline_logs/l$i.log 2>&1
    make $make_options TEST_LEVEL=$i all run >> ../prepare_pipeline_logs/l$i.log 2>&1
    compare expected_output_l$i.pgm output.pgm -metric AE ../prepare_pipeline_logs/diff_l$i.png >> ../prepare_pipeline_logs/l$i.log 2>&1
    mv output.pgm ../prepare_pipeline_logs/output_$i.png
    if [ $? -eq 0 ]; then
        echo -e "Prepare L$i test \e[32mPASSED\e[0m"
    else
        echo -e "Prepare L$i test \e[31mFAILED\e[0m"
        (( status++ ))
    fi
done

exit $status
