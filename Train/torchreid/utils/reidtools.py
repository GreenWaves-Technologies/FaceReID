from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil
import matplotlib.pyplot as plt

import torch
import cv2

from .iotools import mkdir_if_missing


def image_and_black_bg(image_path, res, gray_border, frame=None, eps=20, frame_size=5):
    tmp = cv2.imread(image_path)
    tmp = cv2.resize(tmp, (res, res))
    blck_bg = np.zeros((res + eps, res + eps, 3))
    ch = 0
    if frame == 'green':
        ch = 1
    elif frame == 'red':
        ch = 2
    if frame is not None:
        blck_bg[eps // 2 - frame_size:eps // 2 + res + frame_size, eps // 2 - frame_size:eps //2 + res + frame_size, ch] = 256
    blck_bg[eps // 2:eps //2 + res, eps // 2:eps //2 + res, :] = tmp
    if gray_border:
        blck_bg[:, eps // 2 + res:, :] = 128
    return blck_bg


def draw_mosaic(save_dir, topk=5, res=128, max_items=30):
    overall_mosaic = None
    for idx, folder in enumerate(sorted(os.listdir(save_dir))[:max_items]):
        folder_path = osp.join(save_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        images_list = [os.path.join(folder_path, image_name) for image_name in sorted(os.listdir(folder_path))]
        query_pid = int(sorted(os.listdir(folder_path))[-1].split('_')[3])
        horiz_line = image_and_black_bg(images_list[-1], res, True, )
        for image_path in images_list[:topk]:
            gallery_pid = int(osp.basename(image_path).split('_')[4])
            frame = None
            if gallery_pid == query_pid:
                frame = 'green'
            tmp = image_and_black_bg(image_path, res, False, frame)
            horiz_line = np.hstack((horiz_line, tmp))
        if overall_mosaic is None:
            overall_mosaic = horiz_line
        else:
            overall_mosaic = np.vstack((overall_mosaic, horiz_line))
    cv2.imwrite(osp.join(save_dir, 'mosaic.png'), overall_mosaic)


def visualize_ranked_results(distmat, dataset, save_dir='log/ranked_results', topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_dir))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix, gidx=None):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_pid_' + str(gidx) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)
    counter = 0
    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qdir = osp.join(save_dir, 'query' + str(q_idx + 1).zfill(5)) #osp.basename(qimg_path))
        mkdir_if_missing(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix='query', gidx=qpid)
        if counter > 50:
            break
        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='_gallery', gidx=gpid)
                rank_idx += 1
                if rank_idx > topk:
                    break
        counter += 1
    print("Done")
    draw_mosaic(save_dir)


def distmat_hist(distmat, g_pids, q_pids):
    pos_examples = distmat[g_pids[np.newaxis, :] == q_pids[:, np.newaxis]]
    neg_examples = distmat[g_pids[np.newaxis, :] != q_pids[:, np.newaxis]]
    plt.hist(pos_examples, bins=50, facecolor='g', alpha=0.5)
    plt.hist(neg_examples, bins=50, facecolor='r', alpha=0.5)
    plt.show()


def calc_distmat(qf, gf, distance='cosine'):
    m, n = qf.size(0), gf.size(0)

    if distance == 'l2':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
    elif distance == 'cosine':
        squares = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) * \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = torch.ones((m, n)) - torch.mm(qf, gf.t()) / torch.pow(squares, 0.5)

    return distmat