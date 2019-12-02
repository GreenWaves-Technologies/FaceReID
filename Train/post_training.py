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
from torchreid.utils.quantization import collect_stats, quantize_hook, save_quantized_weights, save_act_hook
from torchreid.utils.convert import convert_to_onnx
from torchreid.utils.inference import inference
from test_on_lfw import roc_auc
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
                              ceil_mode=not args.convert_to_onnx, infer=True,
                              quantized=args.quantization or args.quantized, bits=args.bits,
                              normalize_embeddings=args.normalize_embeddings, normalize_fc=args.normalize_fc)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        # if args.quantized:
        #     num_channels = 3
        #     if args.grayscale:
        #         num_channels = 1
        #     model.eval()
        #     _ = model(torch.rand(1, num_channels, 224, 224))
        # load pretrained weights but ignore layers that don't match in size
        load_weights(model, args.load_weights)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.convert_to_onnx:
        convert_to_onnx(model, args)
        return

    mean = torch.tensor(args.mean)
    std = torch.tensor(args.std)
    scale = 255

    if args.no_normalize and not args.quantized:
        model.conv1.bias.data = model.conv1.bias.data - model.conv1.weight.data[:, 0, 0, 0] * mean / std
        model.conv1.weight.data /= (scale * std)
    print(model.conv1.bias.data, model.conv1.weight.data)
    if args.absorb_bn:
        search_absorbed_bn(model)

    if use_gpu:
        # model = nn.DataParallel(model).cuda()
        model = model.cuda()

    if args.quantization or args.save_quantized_model:
        if args.no_normalize:
            int_input_bits = 15  # input is a tensor filled with 8-bit integers
        else:
            int_input_bits = 2  # can be derived from normalize transform parameters

        if args.quantization:
            hook = partial(quantize_hook, bits=args.bits, int_input_bits=int_input_bits, accum_bits=32)

        handles = []
        for name, module in model.named_modules():
            handles.append(module.register_forward_hook(partial(hook, module_name=name)))

        collect_stats(model, testloader_dict[args.target_names[0]]['query'], use_gpu)
        for handle in handles:  # delete forward hooks
            handle.remove()
        if args.save_quantized_pytorch_model:
            # if use_gpu:
            #     state_dict = model.module.state_dict()
            # else:
            state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                # 'lfw_acc': all_acc,  # rank1 on the last measured dataset!
                # 'epoch': epoch,
                # 'optim': optimizer.state_dict()
            }, False, args.load_weights[:args.load_weights.find('.')] + '_quantized.pth.tar')
            print('quantized model was saved to ', args.load_weights[:args.load_weights.find('.')] + '_quantized.pth.tar')
        if args.save_quantized_model:
            for name, module in model.named_modules():
                save_quantized_weights(module, name, os.path.join(args.save_dir, 'quantized_model'))
        print('Weights were quantized!')

    if args.infer:
        if args.image_path == '':
            raise AttributeError('Image for inference is required')

        handles = []
        for name, module in model.named_modules():
            hook = partial(save_act_hook, save_dir=os.path.join(args.save_dir, 'activation_dump'), name=name)
            handles.append(module.register_forward_hook(hook))
        inference(model, args.image_path, args, use_gpu)

        for handle in handles:  # delete forward hooks
            handle.remove()

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            if 'msceleb' in name.lower():
                continue
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