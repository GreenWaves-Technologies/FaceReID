# Quantization

## Installation and environment setup

1. Clone the repository from Github:

```
    $ sudo pip3 install git+https://github.com/xperience-ai/gap_quantization.git
    $ git clone https://github.com/xperience-ai/gap_quantization.git
    $ cd gap_quantization

```

2. Install the development enviroment:

```
    # create virtual environment using virtualenv (in this example)/pyvenv/conda/etc.
    $ virtualenv -p /usr/bin/python3 ./venv
    $ . venv/bin/activate
    $ pip3 install -r ./requirements-dev.txt

```

## Quantization

1. Create a folder with a subset of the validation dataset (~100 images should be enough) and specify path to it in `examples/quantization_re_id.py`'s cfg:
`"data_source": "<created folder>"`

2. Run quantization script:

```
    $PYTHONPATH=. python examples/quantization_re_id.py --trained-model <path to saved checkpoints of last epoch>
```

Eventually new folder `results` will be created with 26 *.json files of quantized model in it and subfolder `activations_dump` inside.
Directory `results` contains quantized weights and biases.
Directory `activations_dump` contains input and output of each layer. We'll need it to have a possibility to compare results on GAP with desirable.
File `norm_list.h` contains norm parameters of all layers in network, which is responsible for separator position in fixed point calculation.

## Run test on GAP

Now, when you have quantized your model, you can upload weights to GAP and check if everything went right.

1. Copy *.json files from `results` and put it into `ReID-Demo/quantized_model` instead of pre-trained weights.

Copy folder `results/activations_dump` and put it into `tests` instead of existed folder.

Copy file `norm_list.h` and put it into `ReID-Demo` folder.

2. Go to directory `tests` and run
```
    $./test_layers_one_by_one.sh <target platform>
```
Use `-gapoc` as target platform if you're using Gapoc A board and `-gapuino` if you're using Gapuino board.
You will see results of tests on each layer like this:
```
    Layer 0: conv1/input.json => conv1/output.json
    Layer 0: conv1/input.json => conv1/output.json PASSED

```

See [Build and test instructions](./build_test.md) for more detailed model testing.