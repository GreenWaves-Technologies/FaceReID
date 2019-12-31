from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
from collections import defaultdict
import numpy as np
import math
from functools import partial
from tqdm import tqdm
import glog as log

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger
from torchreid.utils.torchtools import count_num_param
from torchreid.utils.reidtools import visualize_ranked_results, distmat_hist, calc_distmat
from torchreid.eval_metrics import test
from torchreid.utils.load_weights import load_weights
from torchreid.utils.absorb_bn import search_absorbed_bn
from torchreid.evaluate_lfw import evaluate, compute_embeddings_lfw


# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args

    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    log_name = 'log_test.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    if not args.convert_to_onnx:  # and not args.infer:
        dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
        trainloader, trainloader_dict, testloader_dict = dm.return_dataloaders()

    num_train_pids = 100

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=num_train_pids, loss={'xent', 'htri'},
                              pretrained=False if args.load_weights else 'imagenet', grayscale=args.grayscale,
                              ceil_mode=not args.convert_to_onnx, infer=True, bits=args.bits,
                              normalize_embeddings=args.normalize_embeddings, normalize_fc=args.normalize_fc, convbn=args.convbn)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        load_weights(model, args.load_weights)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.absorb_bn:
        search_absorbed_bn(model)

    if args.quantization or args.save_quantized_model:
        from gap_quantization.quantization import ModelQuantizer

        if args.quant_data_dir is None:
            raise AttributeError('quant-data-dir argument is required.')

        num_channels = 1 if args.grayscale else 3
        cfg = {
            "bits": args.bits,  # number of bits to store weights and activations
            "accum_bits": 32,  # number of bits to store intermediate convolution result
            "signed": True,  # use signed numbers
            "save_folder": args.save_dir,  # folder to save results
            "data_source": args.quant_data_dir,  # folder with images to collect dataset statistics
            "use_gpu": False,  # use GPU for inference
            "batch_size": 1,
            "num_workers": 0,  # number of workers for PyTorch dataloader
            "verbose": True,
            "save_params": args.save_quantized_model,  # save quantization parameters to the file
            "quantize_forward": True,  # replace usual convs, poolings, ... with GAP-like ones
            "num_input_channels": num_channels,
            "raw_input": args.no_normalize,
            "double_precision": args.double_precision # use double precision convolutions
        }

        model = model.cpu()
        quantizer = ModelQuantizer(model, cfg, dm.transform_test)  # transform test is OK if we use args.no_normalize
        quantizer.quantize_model()                                  # otherwise we need to add QuantizeInput operation

        if args.infer:
            if args.image_path == '':
                raise AttributeError('Image for inference is required')

            quantizer.dump_activations(args.image_path, dm.transform_test,
                                       save_dir=os.path.join(args.save_dir, 'activation_dump'))

    if use_gpu:
        model = nn.DataParallel(model).cuda()


    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            if not 'lfw' in name.lower():
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                distmat = test(args, model, queryloader, galleryloader, use_gpu, return_distmat=True)

                if args.visualize_ranks:
                    visualize_ranked_results(
                        distmat, dm.return_testdataset_by_name(name),
                        save_dir=osp.join(args.save_dir, 'ranked_results', name),
                        topk=20
                    )

            else:
                model.eval()
                same_acc, diff_acc, all_acc, auc, thresh = evaluate(args, dm.lfw_dataset, model, compute_embeddings_lfw,
                                                            args.test_batch_size, verbose=False, show_failed=args.show_failed, load_embeddings=args.load_embeddings)
                log.info('Validation accuracy: {0:.4f}, {1:.4f}'.format(same_acc, diff_acc))
                log.info('Validation accuracy mean: {0:.4f}'.format(all_acc))
                log.info('Validation AUC: {0:.4f}'.format(auc))
                log.info('Estimated threshold: {0:.4f}'.format(thresh))
                #roc_auc(model, '/home/maxim/data/lfw/pairsTest.txt', '/media/slow_drive/cropped_lfw', args, use_gpu)
        return

if __name__ == '__main__':
    main()