#!/bin/bash

if [ "$1" = "-gapoc" ]; then
    MAKEFILE_NAME="Makefile BOARD_NAME=gapoc_a"
elif [ "$1" = "-gapuino" ]; then
    MAKEFILE_NAME="Makefile BOARD_NAME=gapuino"
else
    echo "Target platform is not defined, use -gapoc for Gapoc A board and -gapuino for Gapuino board"
    exit 2
fi

mkdir -p prepare_pipeline_logs

status=0
cd ./prepare_pipeline_test

for i in `seq 0 2`
do
    echo "Prepare L$i test"
    make -f $MAKEFILE_NAME TEST_LEVEL=$i clean > /dev/null 2>&1
    make -f $MAKEFILE_NAME TEST_LEVEL=$i tiler_models > ../prepare_pipeline_logs/l$i.log 2>&1
    make -f $MAKEFILE_NAME TEST_LEVEL=$i -j4 run >> ../prepare_pipeline_logs/l$i.log 2>&1
    cp ./output.pgm ../prepare_pipeline_logs/output_$i.png
    compare expected_output_l$i.pgm output.pgm -metric AE ../prepare_pipeline_logs/diff_l$i.png >> ../prepare_pipeline_logs/l$i.log 2>&1
    if [ "$?" -eq "0" ];
    then
        echo -e "Prepare L$i test \e[32mPASSED\e[0m"
    else
        echo -e "Prepare L$i test \e[31mFAILED\e[0m"
        let "status = $status + 1"
    fi
done

exit $status
