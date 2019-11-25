from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import glog as log

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import choose_losses
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results, distmat_hist
from torchreid.eval_metrics import test
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optimizer
from torchreid.utils.load_weights import load_weights
from torchreid.utils.flops_counter import count_flops
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
    log_name = 'log_train_{}.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S"))
    if args.evaluate:
        log_name.replace('train', 'test')
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print(' '.join(sys.argv))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")

    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    if hasattr(dm, 'lfw_dataset'):
        lfw = dm.lfw_dataset
        print('LFW dataset is used!')
    else:
        lfw = None

    trainloader, trainloader_dict, testloader_dict = dm.return_dataloaders()

    num_train_pids = dm.num_train_pids

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=num_train_pids, loss={'xent', 'htri'},
                              pretrained=False if args.load_weights else 'imagenet', grayscale=args.grayscale,
                              normalize_embeddings=args.normalize_embeddings, normalize_fc=args.normalize_fc, convbn=args.convbn)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    count_flops(model, args.height, args.width, args.grayscale)

    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        load_weights(model, args.load_weights)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint from '{}'".format(args.resume))

        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, checkpoint['rank1']))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        model = model.cuda()

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
                                                            args.test_batch_size, verbose=False, show_failed=args.show_failed)
                log.info('Validation accuracy: {0:.4f}, {1:.4f}'.format(same_acc, diff_acc))
                log.info('Validation accuracy mean: {0:.4f}'.format(all_acc))
                log.info('Validation AUC: {0:.4f}'.format(auc))
                log.info('Estimated threshold: {0:.4f}'.format(thresh))
        return

    criterions = choose_losses(args, dm, model, use_gpu)

    if not args.evaluate and len(criterions) == 0:
        raise AssertionError('No loss functions were chosen!')
    optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))

    if args.load_optim:
        checkpoint = torch.load(args.load_weights)
        optimizer.load_state_dict(checkpoint['optim'])
        print("Loaded optimizer from '{}'".format(args.load_weights))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        param_group['weight_decay'] = args.weight_decay

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    start_time = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    train_writer = SummaryWriter(osp.join(args.save_dir, 'train_log'))
    test_writer = SummaryWriter(osp.join(args.save_dir, 'test_log'))
    print("=> Start training")

    if args.fixbase_epoch > 0:
        print("Train {} for {} epochs while keeping other layers frozen".format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            train(epoch, model, criterions, optimizer, trainloader, use_gpu, train_writer, fixbase=True, lfw=lfw)
            train_time += round(time.time() - start_train_time)

            for name in args.target_names:
                if not 'lfw' in name.lower():
                    print("Evaluating {} ...".format(name))
                    queryloader = testloader_dict[name]['query']
                    galleryloader = testloader_dict[name]['gallery']
                    testloader = testloader_dict[name]['test']
                    criteria = None
                    rank1 = test(args, model, queryloader, galleryloader, use_gpu,
                                 testloader=testloader, criterions=criteria)
                else:
                    model.eval()
                    same_acc, diff_acc, all_acc, auc, thresh = evaluate(args, dm.lfw_dataset, model, compute_embeddings_lfw,
                                                                        args.test_batch_size, verbose=False,
                                                                        show_failed=args.show_failed)
                    print('Validation accuracy: {0:.4f}, {1:.4f}'.format(same_acc, diff_acc))
                    print('Validation accuracy mean: {0:.4f}'.format(all_acc))
                    print('Validation AUC: {0:.4f}'.format(auc))
                    print('Estimated threshold: {0:.4f}'.format(thresh))
                    rank1 = all_acc

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)

    for epoch in range(args.start_epoch, args.max_epoch):
        for criterion in criterions:
            criterion.train_stats.reset()

        start_train_time = time.time()
        train(epoch, model, criterions, optimizer, trainloader, use_gpu, train_writer, lfw=lfw)
        train_time += round(time.time() - start_train_time)
        
        scheduler.step()
        
        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            num_iter = (epoch + 1) * len(trainloader)
            if not args.no_train_quality:
                for name in args.source_names:
                    print("Measure quality on the {} train set...".format(name))
                    queryloader = trainloader_dict[name]['query']
                    galleryloader = trainloader_dict[name]['gallery']
                    rank1 = test(args, model, queryloader, galleryloader, use_gpu)
                    train_writer.add_scalar('rank1/{}'.format(name), rank1, num_iter)

            print("=> Test")
            
            for name in args.target_names:
                if not 'lfw' in name.lower():
                    print("Evaluating {} ...".format(name))
                    queryloader = testloader_dict[name]['query']
                    galleryloader = testloader_dict[name]['gallery']
                    testloader = testloader_dict[name]['test']
                    criteria = criterions
                    if args.no_loss_on_val:
                        criteria = None
                    rank1 = test(args, model, queryloader, galleryloader, use_gpu,
                                 testloader=testloader, criterions=criteria)
                    test_writer.add_scalar('rank1/{}'.format(name), rank1, num_iter)
                    if not args.no_loss_on_val:
                        for criterion in criterions:
                            test_writer.add_scalar('loss/{}'.format(criterion.name), criterion.test_stats.avg, num_iter)
                            criterion.test_stats.reset()
                    ranklogger.write(name, epoch + 1, rank1)
                else:
                    model.eval()
                    same_acc, diff_acc, all_acc, auc, thresh = evaluate(args, dm.lfw_dataset, model, compute_embeddings_lfw,
                                                                        args.test_batch_size, verbose=False,
                                                                        show_failed=args.show_failed)
                    print('Validation accuracy: {0:.4f}, {1:.4f}'.format(same_acc, diff_acc))
                    print('Validation accuracy mean: {0:.4f}'.format(all_acc))
                    print('Validation AUC: {0:.4f}'.format(auc))
                    print('Estimated threshold: {0:.4f}'.format(thresh))
                    test_writer.add_scalar('Accuracy/Val_same_accuracy', same_acc, num_iter)
                    test_writer.add_scalar('Accuracy/Val_diff_accuracy', diff_acc, num_iter)
                    test_writer.add_scalar('Accuracy/Val_accuracy', all_acc, num_iter)
                    test_writer.add_scalar('Accuracy/AUC', auc, num_iter)
                    rank1 = all_acc
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()


            save_dict = {
                'state_dict': state_dict,
                'epoch': epoch,
                'optim': optimizer.state_dict()
            }

            if len(args.target_names):
                save_dict['rank1'] = rank1

            save_checkpoint(save_dict, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    ranklogger.show_summary()


def train(epoch, model, criterions, optimizer, trainloader, use_gpu, train_writer, fixbase=False, lfw=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    if fixbase or args.always_fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        iteration = epoch * len(trainloader) + batch_idx

        if lfw is not None and iteration % args.val_step == 0 and iteration != 0:
            checkpoint_name = osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1)+ '_iter' + str(batch_idx + 1) + '.pth.tar')

            log.info('Evaluating Snapshot: ' + checkpoint_name)
            model.eval()
            same_acc, diff_acc, all_acc, auc, thresh = evaluate(args, lfw, model, compute_embeddings_lfw,
                                                        args.test_batch_size, verbose=False)

            if iteration > 0:
                log.info('Saving Snapshot: ' + checkpoint_name)
                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()

                save_checkpoint({
                    'state_dict': state_dict,
                    'lfw_acc': all_acc,  # rank1 on the last measured dataset!
                    'epoch': epoch,
                    'optim': optimizer.state_dict()
                }, False, checkpoint_name)

            model.train()

            print('Validation accuracy: {0:.4f}, {1:.4f}'.format(same_acc, diff_acc))
            print('Validation accuracy mean: {0:.4f}'.format(all_acc))
            print('Validation AUC: {0:.4f}'.format(auc))
            print('Estimated threshold: {0:.4f}'.format(thresh))
            train_writer.add_scalar('Accuracy/Val_same_accuracy', same_acc, iteration)
            train_writer.add_scalar('Accuracy/Val_diff_accuracy', diff_acc, iteration)
            train_writer.add_scalar('Accuracy/Val_accuracy', all_acc, iteration)
            train_writer.add_scalar('Accuracy/AUC', auc, iteration)
            #exit(-1)

        data_time.update(time.time() - end)

        if fixbase and batch_idx > 100:
            break

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        
        outputs, features = model(imgs)

        losses = torch.zeros([1]).cuda()
        kwargs = {'targets': pids, 'imgs': imgs}
        for criterion in criterions:
            inputs = features
            if criterion.name == 'xent' or 'am':
                inputs = outputs
            loss = criterion.weight * criterion.calc_loss(inputs, **kwargs)
            losses += loss
            if np.isnan(loss.item()):
                logged_value = sys.float_info.max
            else:
                logged_value = loss.item()
            criterion.train_stats.update(logged_value, pids.size(0))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        if (batch_idx + 1) % args.print_freq == 0:
            output_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch + 1, batch_idx + 1, len(trainloader))
            output_string += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
            output_string += 'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'.format(data_time=data_time)
            for criterion in criterions:
                output_string += 'Loss {}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(criterion.name, loss=criterion.train_stats)
                train_writer.add_scalar('loss/{}'.format(criterion.name), criterion.train_stats.val, iteration)
            print(output_string)
        end = time.time()


if __name__ == '__main__':
    main()
