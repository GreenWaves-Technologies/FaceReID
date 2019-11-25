import argparse
import datetime
from functools import partial

import cv2 as cv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as t

from scipy.spatial.distance import cosine
import glog as log
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# from datasets.lfw import LFW
# from utils.utils import load_model_state, get_model_parameters_number, flip_tensor
# from utils.augmentation import ResizeNumpy, CenterCropNumpy, NumpyToTensor
# from utils.face_align import FivePointsAligner
# from model.common import models_backbones


def get_subset(container, subset_bounds):
    """Returns a subset of the given list with respect to the list of bounds"""
    subset = []
    for bound in subset_bounds:
        subset += container[bound[0]: bound[1]]
    return subset


def get_roc(scores_with_gt, n_threshs=400, min_thresh=0., max_thresh=4.):
    """Computes a ROC curve on the LFW dataset"""
    # scores = [item['score'] for item in scores_with_gt]
    # gts = [item['is_same'] for item in scores_with_gt]
    # thresholds = np.linspace(min(scores), max(scores), n_threshs)
    thresholds = np.linspace(min_thresh, max_thresh, n_threshs)

    fp_rates = []
    tp_rates = []

    for threshold in thresholds:
        fp = 0
        tp = 0
        for score_with_gt in scores_with_gt:
            predict_same = score_with_gt['score'] < threshold
            actual_same = score_with_gt['is_same']

            if predict_same and actual_same:
                tp += 1
            elif predict_same and not actual_same:
                fp += 1

        fp_rates.append(float(fp) / len(scores_with_gt) * 2)
        tp_rates.append(float(tp) / len(scores_with_gt) * 2)
    return np.array(fp_rates), np.array(tp_rates)


def get_auc(fprs, tprs):
    """Computes AUC under a ROC curve"""
    sorted_fprs, sorted_tprs = zip(*sorted(zip(*(fprs, tprs))))
    sorted_fprs = list(sorted_fprs)
    sorted_tprs = list(sorted_tprs)
    if sorted_fprs[-1] != 1.0:
        sorted_fprs.append(1.0)
        sorted_tprs.append(sorted_tprs[-1])
    return np.trapz(sorted_tprs, sorted_fprs)


def save_roc(fp_rates, tp_rates, fname):
    assert fp_rates.shape[0] == tp_rates.shape[0]
    with open(fname + '.txt', 'w') as f:
        for i in range(fp_rates.shape[0]):
            f.write('{} {}\n'.format(fp_rates[i], tp_rates[i]))


def load_embedding(file_path):
    emb = []
    f = open(file_path, "rb")
    for _ in range(512):
        emb.append(int.from_bytes(f.read(2), 'little'))
    return torch.FloatTensor(emb)


@torch.no_grad()
def compute_embeddings_lfw(args, dataset, model, batch_size, dump_embeddings,
                           pdist, flipped_embeddings=False, load_embeddings=False):
    """Computes embeddings of all images from the LFW dataset using PyTorch"""
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    scores_with_gt = []
    embeddings = []
    ids = []

    for batch_idx, data in enumerate(tqdm(val_loader, 'Computing embeddings')):
        if not load_embeddings:
            images_1 = data['img1']
            images_2 = data['img2']
            is_same = data['is_same']
            if torch.cuda.is_available():
                images_1 = images_1.cuda()
                images_2 = images_2.cuda()
            emb_1 = model(images_1)
            emb_2 = model(images_2)
            if flipped_embeddings:
                images_1_flipped = flip_tensor(images_1, 3)
                images_2_flipped = flip_tensor(images_2, 3)
                emb_1_flipped = model(images_1_flipped)
                emb_2_flipped = model(images_2_flipped)
                emb_1 = (emb_1 + emb_1_flipped)*.5
                emb_2 = (emb_2 + emb_2_flipped)*.5
        else:
            emb_1 = torch.zeros((0, 512))
            emb_2 = torch.zeros((0, 512))
            is_same = []
            for idx, _ in enumerate(data['path1']):
                exc = False
                try:
                    emb_1_tmp = load_embedding(data['path1'][idx] + '.bin')
                except FileNotFoundError:
                    print(data['path1'][idx] + '.bin')
                    exc = True
                try:
                    emb_2_tmp = load_embedding(data['path2'][idx] + '.bin')
                except FileNotFoundError:
                    print(data['path2'][idx] + '.bin')
                    exc = True
                if exc:
                    continue
                emb_1 = torch.cat((emb_1, emb_1_tmp.unsqueeze_(0)))
                emb_2 = torch.cat((emb_2, emb_2_tmp.unsqueeze_(0)))
                is_same.append(data['is_same'][idx])
        scores = pdist(emb_1, emb_2).data.cpu().numpy()

        for i, _ in enumerate(scores):
            scores_with_gt.append({'score': scores[i], 'is_same': is_same[i], 'idx': batch_idx*batch_size + i})

        if dump_embeddings:
            id0 = data['id0']
            id1 = data['id1']
            ids.append(id0)
            ids.append(id1)
            to_dump_1 = emb_1.data.cpu()
            to_dump_2 = emb_2.data.cpu()
            embeddings.append(to_dump_1)
            embeddings.append(to_dump_2)

    if dump_embeddings:
        total_emb = np.concatenate(embeddings, axis=0)
        total_ids = np.concatenate(ids, axis=0)
        log_path = './logs/{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
        writer = SummaryWriter(log_path)
        writer.add_embedding(torch.from_numpy(total_emb), total_ids)
    print(len(scores_with_gt))
    return scores_with_gt


def compute_embeddings_lfw_ie(args, dataset, model, batch_size=1, dump_embeddings=False,
                              pdist=cosine, flipped_embeddings=False, lm_model=None):
    """Computes embeddings of all images from the LFW dataset using Inference Engine"""
    assert batch_size == 1
    scores_with_gt = []

    for batch_idx, data in enumerate(tqdm(dataset, 'Computing embeddings')):
        images_1 = data['img1']
        images_2 = data['img2']
        if lm_model:
            lm_input_size = tuple(lm_model.get_input_shape()[2:])
            landmarks_1 = lm_model.forward(cv.resize(images_1, lm_input_size)).reshape(-1)
            images_1 = FivePointsAligner.align(images_1, landmarks_1, *images_1.shape[:2], normalize=False, show=False)

            landmarks_2 = lm_model.forward(cv.resize(images_2, lm_input_size)).reshape(-1)
            images_2 = FivePointsAligner.align(images_2, landmarks_2, *images_2.shape[:2], normalize=False)

        is_same = data['is_same']
        emb_1 = model.forward(images_1).reshape(-1)
        emb_2 = model.forward(images_2).reshape(-1)
        score = pdist(emb_1, emb_2)
        scores_with_gt.append({'score': score, 'is_same': is_same, 'idx': batch_idx * batch_size})

    return scores_with_gt


def compute_optimal_thresh(scores_with_gt):
    """Computes an optimal threshold for pairwise face verification"""
    pos_scores = []
    neg_scores = []
    for score_with_gt in scores_with_gt:
        if score_with_gt['is_same']:
            pos_scores.append(score_with_gt['score'])
        else:
            neg_scores.append(score_with_gt['score'])

    hist_pos, bins = np.histogram(np.array(pos_scores), 60)
    hist_neg, _ = np.histogram(np.array(neg_scores), bins)

    intersection_bins = []

    for i in range(1, len(hist_neg)):
        if hist_pos[i - 1] >= hist_neg[i - 1] and 0.05 < hist_pos[i] <= hist_neg[i]:
            intersection_bins.append(bins[i])

    if not intersection_bins:
        intersection_bins.append(0.5)
    return np.mean(intersection_bins)

    # scores = [item['score'] for item in scores_with_gt]
    # gts = [item['is_same'] for item in scores_with_gt]
    #
    # best_acc = 0
    # thresh = 0
    # for score in sorted(scores):
    #     acc = accuracy_score(gts, [item < score for item in scores])
    #     #print(acc, score)
    #     if acc > best_acc:
    #         thresh = score
    #         best_acc = acc
    #
    # return thresh


def evaluate(args, dataset, model, compute_embeddings_fun, val_batch_size=16,
             dump_embeddings=False, roc_fname='', snap_name='', verbose=False, show_failed=False, load_embeddings=False):
    """Computes the LFW score of given model"""
    # if verbose and isinstance(model, torch.nn.Module):
    #     log.info('Face recognition model config:')
    #     log.info(model)
    #     log.info('Number of parameters: {}'.format(get_model_parameters_number(model)))
    model.eval()
    if args.distance == 'cosine':
        dist = lambda x, y: 1. - F.cosine_similarity(x, y)
    elif args.distance == 'l2':
        dist = torch.nn.PairwiseDistance()
    scores_with_gt = compute_embeddings_fun(args, dataset, model, val_batch_size, dump_embeddings, dist,
                                            load_embeddings=load_embeddings)
    num_pairs = len(scores_with_gt)

    subsets = []
    for i in range(10):
        lower_bnd = i * num_pairs // 10
        upper_bnd = (i + 1) * num_pairs // 10
        subset_test = [(lower_bnd, upper_bnd)]
        subset_train = [(0, lower_bnd), (upper_bnd, num_pairs)]
        subsets.append({'test': subset_test, 'train': subset_train})

    same_scores = []
    diff_scores = []
    val_scores = []
    threshs = []
    mean_fpr = np.zeros(400)
    mean_tpr = np.zeros(400)
    failed_pairs = []

    scores = []
    gts = []

    for subset in tqdm(subsets, '{} evaluation'.format(snap_name), disable=not verbose):
        train_list = get_subset(scores_with_gt, subset['train'])
        optimal_thresh = compute_optimal_thresh(train_list)
        threshs.append(optimal_thresh)

        test_list = get_subset(scores_with_gt, subset['test'])
        same_correct = 0
        diff_correct = 0
        pos_pairs_num = neg_pairs_num = len(test_list) // 2

        for score_with_gt in test_list:
            scores.append(1 - score_with_gt['score'])
            gts.append(score_with_gt['is_same'])

            if score_with_gt['score'] < optimal_thresh and score_with_gt['is_same']:
                same_correct += 1
            elif score_with_gt['score'] >= optimal_thresh and not score_with_gt['is_same']:
                diff_correct += 1

            if score_with_gt['score'] >= optimal_thresh and score_with_gt['is_same']:
                failed_pairs.append(score_with_gt)
            if score_with_gt['score'] < optimal_thresh and not score_with_gt['is_same']:
                failed_pairs.append(score_with_gt)

        same_scores.append(float(same_correct) / pos_pairs_num)
        diff_scores.append(float(diff_correct) / neg_pairs_num)
        val_scores.append(0.5*(same_scores[-1] + diff_scores[-1]))

    for subset in tqdm(subsets, '{} evaluation'.format(snap_name), disable=not verbose):
        test_list = get_subset(scores_with_gt, subset['test'])
        fprs, tprs = get_roc(test_list, mean_fpr.shape[0], min(scores), max(scores))
        mean_fpr = mean_fpr + fprs
        mean_tpr = mean_tpr + tprs

    #print(mean_fpr)
    mean_fpr /= 10
    mean_tpr /= 10

    if roc_fname:
        save_roc(mean_tpr, mean_fpr, roc_fname)

    same_acc = np.mean(same_scores)
    diff_acc = np.mean(diff_scores)
    overall_acc = np.mean(val_scores)
    auc = get_auc(mean_fpr, mean_tpr)

    # while far < FAR_thresh:
    #     far = (i - sum(sorted_by_score[:i])) / num_false_examples
    #     tar = sum(sorted_by_score[:i]) / num_true_examples
    #     i += 1
    # print('TAR: {}, FAR: {}'.format(tar, far))
    #
    # threshold = optimal_threshold
    # accuracy = sum([(score > threshold) == true_val for score, true_val in zip(y_score, y_true)]) / len(y_true)
    # print("Accuracy with optimal threshold: ", accuracy)

    if args.distmat_hist:
        same_pairs_scores = [score for score, gt in zip(scores, gts) if gt]
        diff_pairs_scores = [score for score, gt in zip(scores, gts) if not gt]
        plt.hist(same_pairs_scores, bins=50, facecolor='g', alpha=0.5)
        plt.hist(diff_pairs_scores, bins=50, facecolor='r', alpha=0.5)
        plt.show()

    if show_failed:
        log.info('Number of misclassified pairs: {}'.format(len(failed_pairs)))
        dir_for_failed_pairs = os.path.join(args.save_dir, 'failed_pairs')
        os.makedirs(dir_for_failed_pairs, exist_ok=True)
        for pair in failed_pairs:
            dataset.show_item(dir_for_failed_pairs, pair['idx'], pair['is_same'])

    if verbose:
        log.info('Accuracy/Val_same_accuracy mean: {0:.4f}'.format(same_acc))
        log.info('Accuracy/Val_diff_accuracy mean: {0:.4f}'.format(diff_acc))
        log.info('Accuracy/Val_accuracy mean: {0:.4f}'.format(overall_acc))
        log.info('Accuracy/Val_accuracy std dev: {0:.4f}'.format(np.std(val_scores)))
        log.info('AUC: {0:.4f}'.format(auc))
        log.info('Estimated threshold: {0:.4f}'.format(np.mean(threshs)))
        log.info(threshs)

    # roc_auc = roc_auc_score(gts, scores)
    # log.info('ROC AUC: {0:.4f}'.format(roc_auc))

    return same_acc, diff_acc, overall_acc, auc, np.mean(threshs)
