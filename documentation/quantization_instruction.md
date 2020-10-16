# Quantization

## Installation and environment setup

1. Clone the repository from GitHub:

```
    $ sudo pip3 install git+https://github.com/xperience-ai/gap_quantization.git
    $ git clone https://github.com/xperience-ai/gap_quantization.git
    $ cd gap_quantization
```

2. Install the development environment:

```
    # create virtual environment using virtualenv (in this example)/pyvenv/conda/etc.
    $ virtualenv -p /usr/bin/python3 ./venv
    $ . venv/bin/activate
    $ pip3 install -r ./requirements.txt
```

## Quantization

1. Create a folder with a subset of the validation dataset (~100 images should be enough).

2. Run quantization script:

`python post_training.py -a squeezenet1_1 --grayscale --height 128 --width 128 --target-names lfw --load-weights <path to the trained model> --distance l2 --no-normalize --landmarks-path <path to the txt file with landmarks> --convbn --save-dir <directory to save the results>  --quantization --bits 16 --quant-data-dir <directory with images for quantization> --save-quantized-model --infer --image-path <path to the image for inference>`

To measure the quality with quantized model: `python post_training.py -a squeezenet1_1 --grayscale --height 128 --width 128 --target-names lfw --load-weights <path to the trained model> --distance l2 --no-normalize --landmarks-path <path to the txt file with landmarks> --convbn --save-dir <directory to save the results>  --quantization --bits 16 --quant-data-dir <directory with images for quantization> --evaluate`

This mode of quantization can also be used for optimal threshold measurement. After running quantization with `--evaluate` key you will see `Optimal threshold: ...` on screen. This number should be squared and pushed in `ReID-Demo/setup.h` file as `REID_L2_THRESHOLD` and `STRANGER_L2_THRESHOLD`.

Eventually new folder `results` will be created with 26 *.json files of quantized model in it and subfolder `activations_dump` inside.
Directory `results` contains quantized weights and biases.
Directory `activations_dump` contains input and output of each layer. We'll need it to have a possibility to compare results on GAP with desirable.
File `layer_params_quant.h` contains quantization parameters of all layers in network.

## Run test on GAP

Now, when you have quantized your model, you can upload weights to GAP and check if everything went right.

1. Copy *.json files from `results` and put it into `ReID-Demo/quantized_model` instead of pre-trained weights.

Copy folder `results/activations_dump` and put it into `tests` instead of existed folder.

Copy file `layer_params_quant.h` and put it into `ReID-Demo` folder.


2. Go to directory `tests` and run
```
    $ ./test_layers_one_by_one.sh <target platform>
```
Use `-gapoc` as target platform if you're using Gapoc A board and `-gapuino` if you're using Gapuino board.
You will see results of tests on each layer like this:

```
    Layer 0: conv1.0/input.json => conv1.0/output.json
    Layer 0: conv1.0/input.json => conv1.0/output.json PASSED
```

See [Build and test instructions](./build_test.md) for more detailed model testing.
