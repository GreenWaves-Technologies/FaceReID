import argparse
import numpy as np
import os

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', default='/media/slow_drive/VGGFace2/val', help='folder with images')
    parser.add_argument('-s', '--save-dir', default='/media/slow_drive/VGGFace2/val_cropped', help='folder to save crops')
    parser.add_argument('-p', '--prototxt-path', default='face_detector/deploy.prototxt', help='model prototxt path')
    parser.add_argument('-w', '--weights-path', default='face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel',
                        help='model weights path')
    parser.add_argument('--conf-thresh', type=float, default=0.1, help='threshold for confidences')
    return parser.parse_args()


def image_infer(img, net):
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 177, 123])
    net.setInput(blob)
    detections = net.forward()
    return detections


def most_conf_detection(detections, class_id=None):
    confs = detections[0, 0, :, 2]
    if class_id is not None:
        confs[np.isclose(detections[0, 0, :, 1], class_id) == False] = -1
        confs[np.any(detections[0, 0, :, 3:7] > 1, axis=1)] = -1
    return detections[0, 0, np.argmax(confs), 3:7], np.max(confs)


def main():
    args = parse_args()

    root_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    net = cv2.dnn.readNetFromCaffe(args.prototxt_path, args.weights_path)

    for fold in tqdm(os.listdir(root_dir)):
        if not os.path.isdir(os.path.join(root_dir, fold)):
            continue
        os.makedirs(os.path.join(save_dir, fold), exist_ok=True)
        for im_name in os.listdir(os.path.join(root_dir, fold)):
            image_path = os.path.join(root_dir, fold, im_name)
            img = cv2.imread(image_path)
            h, w, _ = img.shape
            detections = image_infer(img, net)
            box, conf = most_conf_detection(detections, 1)
            if conf > args.conf_thresh:
                face = (box * np.array([w, h, w, h])).astype('int')
                img_to_save = img[face[1]:face[3], face[0]:face[2], :]
            else:
                img_to_save = img
            cv2.imwrite(os.path.join(save_dir, fold, im_name), img_to_save)


if __name__ == '__main__':
    main()
