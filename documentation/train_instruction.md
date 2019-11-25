# Train

## Installation and enviroment setup

Then download and install CUDA following the official instructions on the website:

`https://docs.nvidia.com/cuda/cuda-installation-guide-linux/`

Then you need to install Python. We recommend to use virtual enviroment to avoid possible problems with incompatibility of different versions:

```
    $ sudo apt install virtualenv
    $ virtualenv -p /usr/bin/python3 <name of your virtual enviroment>
    $ source <name of your virtual enviroment>/bin/activate
```

Install dependencies by command:

```
    $ cd Train
    $ pip install -r requirements.txt
```

Install the Cython-based evaluation toolbox:

```
    $ cd Train/torchreid/eval_cylib
    $ make
```
As a result, `eval_metrics_cy.so` is generated under the same folder. To test if the toolbox is installed successfully, run

```
    $ python test_cython.py
```

## Datasets

You can use the following datasets to train and test the model:

`vggface2_train` : `http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_train.tar.gz`

`lfw` : `http://vis-www.cs.umass.edu/lfw/lfw.tgz`

When datasets is downloaded, unarchive it.

It is better to use cropped images for model training. First, you'll need to download and unarchive directory with crop boxes:
`www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/bb_landmark.tar.gz`

Then you can crop train images:

```
    $ cd Train
    $ python crop_boxes.py -b <path to downloaded cropboxes>/loose_bb_train.csv -i <path to vggface2_train> -s <path to save cropped images>
```

As the amount of data is large, it may be reasonable to store it somewhere outside of project. In this case, create symbolic links for more convenient data usage:

```
    $ cd Train
    $ mkdir data
    $ cd data
    $ ln -s <path to lfw> lfw
    $ mkdir vggface2
    $ cd vggface2
    $ ln -s <path to vggface2_train cropped> train
```
Also you'll need to download related `.txt` files:

`train_list.txt` : `http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/train_list.txt`

and

`landmarks` : `https://github.com/clcarwin/sphereface_pytorch/blob/master/data/lfw_landmark.txt`

`pairs` : `http://vis-www.cs.umass.edu/lfw/pairs.txt`

and put them to directories `vggface2` and `lfw` respectively.

## Train

Training methods are implemented in `train_imgreid.py`
Input arguments for training script are unified in [args.py](args.py).

To train an image-reid model, go to `Train` directory.

Create folder, in which checkpoints wil be added. They will be saved every 200 iterations and the size of each folder is 62 Mb. So, plese, be prepared, that in the end of 60 epoch checkpoints folder size will be about 75 Gb.

Now you can run the script:

```bash
python train_imgreid.py \
-a squeezenet1_1 \ # network architecture
--grayscale \
--height 128 \ # image height
--width 128 \ # image width
--no-loss-on-val \
--no-train-quality \
--source-names vggface2 \ # source dataset for training
--target-names lfw \ # target dataset for test
--save-dir <path to save checkpoints> \ # where to save the log and models
--train-batch-size 768 \
--eval-freq 1 \ # evaluation frequency
--mean 0.449 \
--std 0.225 \
--distance l2 \
--xent-loss xent \ # use cross entropy loss
--gpu-devices 0 \ # gpu device ids for CUDA_VISIBLE_DEVICES
--landmarks-path data/LFW/landmarks.txt \
--euclid-loss lifted \ # what euclidean-based loss should be used. Possible options: triplet or lifted
--train-sampler RandomIdentitySampler # sampler for trainloader
```

There are some key points will be displayed before training:
```
=> Initializing TRAIN (source) datasets
=> ImageFolder dataset loaded
=> Initializing TEST (target) datasets
LFW dataset is used!
=> Start training
```
Then every 10 iteration table with Epoch, Iteration, Time, Data, Loss xent and Loss euclid values will be displayed.

And every 100 iteration of epoch snapshot will be saved in checkpoint directory and Accuracy and Estimated threshold will be displayed:
```
Computing embeddings: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61/61 [00:11<00:00,  5.24it/s]
6001
I1025 10:13:09.084388 20099 train_imgreid.py:282] Saving Snapshot: ../media/data/checkpoints_new/checkpoint_ep1_iter201.pth.tar
Validation accuracy: 0.7257, 0.7227
Validation accuracy mean: 0.7242
Validation AUC: 0.0000
Estimated threshold: 3.3444

```
