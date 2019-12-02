import os.path as osp
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from PIL import Image
from sklearn.model_selection import StratifiedKFold


class FivePointsAligner():
    """This class performs face alignmet by five reference points"""
    ref_landmarks = np.array([30.2946 / 96, 51.6963 / 112,
                              65.5318 / 96, 51.5014 / 112,
                              48.0252 / 96, 71.7366 / 112,
                              33.5493 / 96, 92.3655 / 112,
                              62.7299 / 96, 92.2041 / 112], dtype=np.float64).reshape(5, 2)
    @staticmethod
    def align(img, landmarks, d_size=(400, 400), normalized=False, show=False):
        """Transforms given image in such a way that landmarks are located near ref_landmarks after transformation"""
        assert len(landmarks) == 10
        assert isinstance(img, np.ndarray)
        landmarks = np.array(landmarks).reshape(5, 2)
        dw, dh = d_size

        keypoints = landmarks.copy().astype(np.float64)
        if normalized:
            keypoints[:, 0] *= img.shape[1]
            keypoints[:, 1] *= img.shape[0]

        keypoints_ref = np.zeros((5, 2), dtype=np.float64)
        keypoints_ref[:, 0] = FivePointsAligner.ref_landmarks[:, 0] * dw
        keypoints_ref[:, 1] = FivePointsAligner.ref_landmarks[:, 1] * dh

        transform_matrix = transformation_from_points(keypoints_ref, keypoints)
        output_im = cv.warpAffine(img, transform_matrix, d_size, flags=cv.WARP_INVERSE_MAP)

        if show:
            tmp_output = output_im.copy()
            for point in keypoints_ref:
                cv.circle(tmp_output, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            for point in keypoints:
                cv.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            img = cv.resize(img, d_size)
            cv.imshow('source/warped', np.hstack((img, tmp_output)))
            cv.waitKey()

        return output_im


def transformation_from_points(points1, points2):
    """Builds an affine transformation matrix form points1 to points2"""
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    u, _, vt = np.linalg.svd(np.matmul(points1.T, points2))
    r = np.matmul(u, vt).T

    return np.hstack(((s2 / s1) * r, (c2.T - (s2 / s1) * np.matmul(r, c1.T)).reshape(2, -1)))


class LFW(Dataset):
    """LFW Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, pairs_path, landmark_file_path, transform=None):
        self.pairs_path = pairs_path
        self.images_root_path = images_root_path
        self.landmark_file_path = landmark_file_path
        self.use_landmarks = len(self.landmark_file_path) > 0
        if self.use_landmarks:
            self.landmarks = self._read_landmarks()
        self.pairs = self._read_pairs()
        self.transform = transform

    def _read_landmarks(self):
        """Reads landmarks of the dataset"""
        landmarks = {}
        with open(self.landmark_file_path, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                key = sp[0][sp[0].rfind('/')+1:]
                landmarks[key] = [[int(sp[i]), int(sp[i+1])] for i in range(1, 11, 2)]

        return landmarks

    def _read_pairs(self):
        """Reads annotation of the dataset"""
        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:  # skip header
                pair = line.strip().split()
                pairs.append(pair)

        file_ext = 'jpg'
        lfw_dir = self.images_root_path
        path_list = []

        for pair in pairs:
            if len(pair) == 3:
                path0 = osp.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                id0 = pair[0]
                path1 = osp.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                id1 = pair[0]
                issame = True
            elif len(pair) == 4:
                path0 = osp.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                id0 = pair[0]
                path1 = osp.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                id1 = pair[0]
                issame = False

            path_list.append((path0, path1, issame, id0, id1))

        # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # dataset = []
        # for _, test_index in skf.split(np.zeros_like(path_list), [item[2] for item in path_list]):
        #     for idx in test_index:
        #         dataset.append(path_list[idx])

        # return dataset
        return path_list

    def _load_img(self, img_path):
        """Loads an image from dist, then performs face alignment and applies transform"""
        img = Image.open(img_path).convert('RGB')

        if self.use_landmarks:
            landmarks = np.array(self.landmarks[img_path[img_path.rfind('/')+1:]]).reshape(-1)
            img = FivePointsAligner.align(np.array(img), landmarks, show=False)
            img = Image.fromarray(img)

        if self.transform is None:
            return img

        return self.transform(img)

    def show_item(self, folder, index, is_same):
        """Saves a pair with a given index to disk"""
        path_1, path_2, _, _, _ = self.pairs[index]
        img1 = cv.imread(path_1)
        img2 = cv.imread(path_2)
        text = 'same'
        if not is_same:
            text = 'diff'

        if self.use_landmarks:
            landmarks1 = np.array(self.landmarks[path_1[path_1.rfind('/')+1:]]).reshape(-1)
            landmarks2 = np.array(self.landmarks[path_2[path_2.rfind('/')+1:]]).reshape(-1)
            img1 = FivePointsAligner.align(img1, landmarks1)
            img2 = FivePointsAligner.align(img2, landmarks2)
        else:
            img1 = cv.resize(img1, (400, 400))
            img2 = cv.resize(img2, (400, 400))
        cv.imwrite(osp.join(folder, 'misclassified_{}_{}.jpg'.format(text, index)), np.hstack([img1, img2]))

    def __getitem__(self, index):
        """Returns a pair of images and similarity flag by index"""
        (path_1, path_2, is_same, id0, id1) = self.pairs[index]
        
        try:
            img1, img2 = self._load_img(path_1), self._load_img(path_2)
            return {'img1': img1, 'img2': img2, 'is_same': is_same, 'id0': id0, 'id1': id1}
        except FileNotFoundError:
            return {'is_same': is_same, 'id0': id0, 'id1': id1, 'path1': path_1, 'path2': path_2}

    def __len__(self):
        """Returns total number of samples"""
        return len(self.pairs)
