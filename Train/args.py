import argparse
import os
import sys


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default='data',
                        help="root path to data directory")
    parser.add_argument('-s', '--source-names', type=str, nargs='+', default=[],
                        help="source datasets (delimited by space)")
    parser.add_argument('-t', '--target-names', type=str, nargs='+', default=[],
                        help="target datasets (delimited by space)")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (tips: 4 or 8 times number of gpus)")
    parser.add_argument('--height', type=int, default=128,
                        help="height of an image")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image")
    parser.add_argument('--split-id', type=int, default=0,
                        help="split index (note: 0-based)")
    parser.add_argument('--train-sampler', type=str, default='',
                        help="sampler for trainloader")
    parser.add_argument('--grayscale', action='store_true',
                        help="use grayscale image")
    parser.add_argument('--no-normalize', action='store_true',
                        help="don\'t normalize image")
    parser.add_argument('--num-test-ids', type=int, default=0,
                        help="Number of unique objects that are used for validation (for ReIDImageFolder dataset)"
                             "All are used by default")
    parser.add_argument('--gallery-items', type=int, default=20,
                        help="Maximum number of gallery images for each query (for ReIDImageFolder dataset")
    parser.add_argument('--clear-val', action='store_true',
                        help="clear val")
    parser.add_argument('--mean', nargs='+', type=float, default=[0.485, 0.456, 0.406],
                        help="Mean of the image")
    parser.add_argument('--std', nargs='+', type=float, default=[0.229, 0.224, 0.225],
                        help="Standard deviation of the image")
    # ************************************************************
    # LFW-specific settings
    # ************************************************************
    parser.add_argument('--val-step', type=int, default=200,
                        help="step for validation on LFW")
    parser.add_argument('--show-failed', action='store_true', help="show misclassified LFW pairs")
    parser.add_argument('--distmat-hist', action='store_true',
                        help='Distances histogram for elements in validation set')

    # ************************************************************
    # Video datasets
    # ************************************************************
    parser.add_argument('--seq-len', type=int, default=15,
                        help="number of images to sample in a tracklet")
    parser.add_argument('--sample-method', type=str, default='evenly',
                        help="how to sample images from a tracklet")
    parser.add_argument('--pool-tracklet-features', type=str, default='avg', choices=['avg', 'max'],
                        help="how to pool features over a tracklet (for video reid)")
    
    # ************************************************************
    # CUHK03-specific setting
    # ************************************************************
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help="use labeled images, if false, use detected images")
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help="use classic split by Li et al. CVPR'14")
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help="use cuhk03's metric for evaluation")
    
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', default=0.0003, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay")   
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="momentum factor for sgd and rmsprop")
    parser.add_argument('--sgd-dampening', default=0, type=float,
                        help="sgd's dampening for momentum")
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help="whether to enable sgd's Nesterov momentum")
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float,
                        help="rmsprop's smoothing constant")
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help="exponential decay rate for adam's first moment")
    parser.add_argument('--adam-beta2', default=0.999, type=float,
                        help="exponential decay rate for adam's second moment")
    
    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', default=60, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful when restart)")
    parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                        help="stepsize to decay learning rate")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")

    parser.add_argument('--train-batch-size', default=32, type=int,
                        help="training batch size")
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help="test batch size")
    
    parser.add_argument('--always-fixbase', action='store_true',
                        help="always fix base network and only train specified layers")
    parser.add_argument('--fixbase-epoch', type=int, default=0,
                        help="how many epochs to fix base network (only train randomly initialized classifier)")
    parser.add_argument('--open-layers', type=str, nargs='+', default=['classifier'],
                        help="open specified layers for training while keeping others frozen")

    # ************************************************************
    # Loss-specific settings
    # ************************************************************
    parser.add_argument('--label-smooth', action='store_true',
                        help="use label smoothing regularizer in cross entropy loss")
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet and lifted losses")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")
    parser.add_argument('--xent-loss', type=str,
                        help="use cross entropy loss")
    parser.add_argument('--euclid-loss', choices=['triplet', 'lifted'],
                        help="what euclidean-based loss should be used. Possible options: triplet or lifted")
    parser.add_argument('--face-loss', choices=['arc', 'sphere', 'cos'],
                        help="what face loss should be used. Possible options: arc, cos or sphere")
    parser.add_argument('--weight-xent', type=float, default=1,
                        help="weight to balance cross entropy loss")
    parser.add_argument('--weight-euclid', type=float, default=1,
                        help="weight to balance euclidean loss (triplet or lifted)")
    parser.add_argument('--weight-face', type=float, default=1,
                        help="weight to balance (Sphere, Cos, Arc)Face loss")
    parser.add_argument('--weight-distill', type=float, default=1,
                        help="weight for distillation loss")

    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('-ta', '--teacher-arch', type=str)

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help="load pretrained weights but ignore layers that don't match in size")
    parser.add_argument('--load-optim', action='store_true',
                        help="load optimizer from pretrained checkpoint")
    parser.add_argument('--load-teacher-weights', type=str, default='',
                        help="load teacher weights")
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluate only")
    parser.add_argument('--eval-freq', type=int, default=-1,
                        help="evaluation frequency (set to -1 to test only in the end)")
    parser.add_argument('--start-eval', type=int, default=0,
                        help="start to evaluate after a specific epoch")
    
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--normalize-embeddings', action='store_true', help='Normalize features vector on the output of the network')
    parser.add_argument('--normalize-fc', action='store_true', help='Normalize weights ofthe classifier after every forward pass'
                                                                    'Usually applied during training with softmax-based loss functions')
    parser.add_argument('--no-train-quality', action='store_true',
                        help="don\'t measure quality on the train set")
    parser.add_argument('--no-loss-on-val', action='store_true',
                        help="don\'t measure loss on the validation set")
    parser.add_argument('--load-embeddings', action='store_true', help='To load computed embeddings')
    parser.add_argument('--landmarks-path', type=str, default='', help='Path to the landmarks file')
    parser.add_argument('--print-freq', type=int, default=10,
                        help="print frequency")
    parser.add_argument('--seed', type=int, default=1,
                        help="manual seed")
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--save-dir', type=str, default='test',
                        help="path to save log and model weights")
    parser.add_argument('--use-cpu', action='store_true',
                        help="use cpu")
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help="use available gpus instead of specified devices (useful when using managed clusters)")
    parser.add_argument('--visualize-ranks', action='store_true',
                        help="visualize ranked results, only available in evaluation mode")
    parser.add_argument('--convert-to-onnx', action='store_true',
                        help="Convert to ONNX")
    parser.add_argument('--absorb-bn', action='store_true')
    parser.add_argument('--quantization', action='store_true', help='Emulate quantized 16 bits inference')
    parser.add_argument('--quantized', action='store_true', help='Flag that loaded model is already quantized')
    parser.add_argument('--bits', type=int, default=16, help='Number of bits to store quantized values')
    parser.add_argument('--infer', action='store_true', help='run inference of the network')
    parser.add_argument('--image-path', type=str, default='face.jpg', help='Image to run inference on')
    parser.add_argument('--save-quantized-model', action='store_true', help='Save quantized weights')
    parser.add_argument('--save-quantized-pytorch-model', action='store_true', help='Save quantized weights in pth file')
    parser.add_argument('--distance', type=str, choices=['l2', 'cosine'], default='l2', help='Distance to use')
    parser.add_argument('--convbn', action='store_true',
                        help='Normalize features vector on the output of the network')
    return parser


def image_dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'num_instances': parsed_args.num_instances,
        'cuhk03_labeled': parsed_args.cuhk03_labeled,
        'cuhk03_classic_split': parsed_args.cuhk03_classic_split,
        'grayscale': parsed_args.grayscale,
        'no_normalize': parsed_args.no_normalize,
        'quantization': parsed_args.quantization,
        'bits': parsed_args.bits,
        'num_test_ids': parsed_args.num_test_ids,
        'gallery_items': parsed_args.gallery_items,
        'clear_val': parsed_args.clear_val,
        'mean': parsed_args.mean,
        'std': parsed_args.std,
        'landmarks_path': parsed_args.landmarks_path
    }


def video_dataset_kwargs(parsed_args):
    """
    Build kwargs for VideoDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'seq_len': parsed_args.seq_len,
        'sample_method': parsed_args.sample_method
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizer.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2
    }