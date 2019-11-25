import argparse
import os

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bbox-file', type=str, required=True, help="csv file with bounding boxes")
    parser.add_argument('-i', '--images-dir', type=str, required=True, help="path to the directory with images")
    parser.add_argument('-s', '--save-dir', type=str, required=True, help="path to the directory to save cropped images")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.bbox_file) as f:
        data = f.read().splitlines()

    for elem in tqdm(data[1:]):
        name_and_box = elem.split(',')
        x, y, w, h = list(map(int, name_and_box[1:]))
        name = name_and_box[0].strip('\"') + '.jpg'
        img = cv2.imread(os.path.join(args.images_dir, name))
        os.makedirs(os.path.join(args.save_dir, name.split('/')[0]), exist_ok=True)
        crop = img[max(0, y):min(y+h, img.shape[0]), max(x, 0):min(x+w, img.shape[1]), :]
        cv2.imwrite(os.path.join(args.save_dir, name), crop)


if __name__ == '__main__':
    main()