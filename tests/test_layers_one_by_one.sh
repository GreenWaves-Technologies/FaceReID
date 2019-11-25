#!/bin/bash

layer_inputs=(\
conv1/input.json \
features.0/input.json \
features.3.squeeze/input.json \
features.3.expand1x1/input.json \
features.3.expand3x3/input.json \
features.4.squeeze/input.json \
features.4.expand1x1/input.json \
features.4.expand3x3/input.json \
features.6.squeeze/input.json \
features.6.expand1x1/input.json \
features.6.expand3x3/input.json \
features.7.squeeze/input.json \
features.7.expand1x1/input.json \
features.7.expand3x3/input.json \
features.9.squeeze/input.json \
features.9.expand1x1/input.json \
features.9.expand3x3/input.json \
features.10.squeeze/input.json \
features.10.expand1x1/input.json \
features.10.expand3x3/input.json \
features.11.squeeze/input.json \
features.11.expand1x1/input.json \
features.11.expand3x3/input.json \
features.12.squeeze/input.json \
features.12.expand1x1/input.json \
features.12.expand3x3/input.json \
global_avgpool/input.json )

layer_outputs=(\
conv1/output.json \
features.2/output.json \
features.3.squeeze_activation/output.json \
features.3.expand1x1_activation/output.json \
features.3.expand3x3_activation/output.json \
features.4.squeeze_activation/output.json \
features.4.maxpool1x1/output.json \
features.4.maxpool3x3/output.json \
features.6.squeeze_activation/output.json \
features.6.expand1x1_activation/output.json \
features.6.expand3x3_activation/output.json \
features.7.squeeze_activation/output.json \
features.7.maxpool1x1/output.json \
features.7.maxpool3x3/output.json \
features.9.squeeze_activation/output.json \
features.9.expand1x1_activation/output.json \
features.9.expand3x3_activation/output.json \
features.10.squeeze_activation/output.json \
features.10.expand1x1_activation/output.json \
features.10.expand3x3_activation/output.json \
features.11.squeeze_activation/output.json \
features.11.expand1x1_activation/output.json \
features.11.expand3x3_activation/output.json \
features.12.squeeze_activation/output.json \
features.12.expand1x1_activation/output.json \
features.12.expand3x3_activation/output.json \
global_avgpool/output.json )

let "layers_count = ${#layer_inputs[@]} - 1"

if [ ${#layer_inputs[@]} -ne ${#layer_outputs[@]} ]; then
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

mkdir -p single_logs
echo "Generating Model"

make -C ../ReID-Demo clean > ./single_logs/model_generaition.log 2>&1
make -j8 -C ../ReID-Demo reid_model >> ./single_logs/model_generaition.log 2>&1 # to generate layers blobs for GAP

echo "Input blob; Output blob; Found outliers; Max diff; Diff summary;" > single_layer_test_summary.csv
cd single_layer_test

status=0

for i in $(seq 0 $layers_count)
do
    echo "Layer $i: ${layer_inputs[$i]} => ${layer_outputs[$i]}"
    make -f $MAKEFILE_NAME clean > /dev/null 2>&1
    rm -rf ./input.bin ./output.bin
    ../../scripts/json2bin.py ../activations_dump/${layer_inputs[$i]} ./input.bin
    make -j4 -f $MAKEFILE_NAME TEST_LAYER_INDEX=$i tiler_models > ../single_logs/$i.stdout.log 2>&1
    make -j4 -f $MAKEFILE_NAME TEST_LAYER_INDEX=$i run >> ../single_logs/$i.stdout.log 2>&1
    echo -n "${layer_inputs[$i]}; ${layer_outputs[$i]}; " >> ../single_layer_test_summary.csv
    ../../scripts/compareWithBin.py ../activations_dump/${layer_outputs[$i]} ./output.bin >> ../single_layer_test_summary.csv
    if [ $? -ne 0 ]; then
        echo ";" >> ../single_layer_test_summary.csv
        echo -e "Layer $i: ${layer_inputs[$i]} => ${layer_outputs[$i]} \e[31mFAILED\e[0m"
        let "status = $status + 1"
    else
        echo -e "Layer $i: ${layer_inputs[$i]} => ${layer_outputs[$i]} \e[32mPASSED\e[0m"
    fi
    mv ./delta.csv ../single_logs/$i.delta.csv
    mv ./output.bin ../single_logs/$i.output.bin
done

cd ..

exit $status
