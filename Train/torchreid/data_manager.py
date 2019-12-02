from __future__ import absolute_import
from __future__ import print_function

import os
import random
import copy

from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset, VideoDataset
from .datasets import init_imgreid_dataset, init_vidreid_dataset
from .datasets.lfw import LFW
from .transforms import build_transforms
from .samplers import RandomIdentitySampler


class BaseDataManager(object):

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.trainloader_dict, self.testloader_dict

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):
    """
    Image-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root,
                 split_id=0,
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 num_instances=4, # number of instances per identity (for RandomIdentitySampler)
                 query_frac=0.2,
                 cuhk03_labeled=False, # use cuhk03's labeled or detected images
                 cuhk03_classic_split=False, # use cuhk03's classic split or 767/700 split
                 landmarks_path='',
                 **kwargs):
        super(ImageDataManager, self).__init__()
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.num_instances = num_instances
        self.cuhk03_labeled = cuhk03_labeled
        self.cuhk03_classic_split = cuhk03_classic_split
        self.pin_memory = True if self.use_gpu else False

        # Build train and test transform functions
        transform_train = build_transforms(self.height, self.width, is_train=True, **kwargs)
        transform_test = build_transforms(self.height, self.width, is_train=False, **kwargs)

        print("=> Initializing TRAIN (source) datasets")
        self.train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        def test_dataloader(data_source): return DataLoader(
            ImageDataset(data_source, transform=transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.pin_memory, drop_last=False)

        self.trainloader_dict = {name: {'query': None, 'gallery': None} for name in self.source_names}
        self.traindataset_dict = {name: {'query': None, 'gallery': None} for name in self.source_names}

        for name in self.source_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, target='train', split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )

            shuffled_dataset = copy.deepcopy(dataset.train)
            random.shuffle(shuffled_dataset)
            num_queries = int(len(dataset.train) * query_frac)

            self.traindataset_dict[name]['query'] = shuffled_dataset[:num_queries]
            self.traindataset_dict[name]['gallery'] = shuffled_dataset[num_queries:]

            self.trainloader_dict[name]['query'] = test_dataloader(self.traindataset_dict[name]['query'])
            self.trainloader_dict[name]['gallery'] = test_dataloader(self.traindataset_dict[name]['gallery'])

            for img_path, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                self.train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        if self.train_sampler == 'RandomIdentitySampler':
            def sampler(data_source): return RandomIdentitySampler(data_source, self.train_batch_size, self.num_instances)
            shuffle = False
        else:
            def sampler(data_source): return None
            shuffle = True
        self.trainloader = DataLoader(
            ImageDataset(self.train, transform=transform_train),
            sampler=sampler(self.train),
            batch_size=self.train_batch_size, shuffle=shuffle, num_workers=self.workers,
            pin_memory=self.pin_memory, drop_last=True
        )

        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {'query': None, 'gallery': None, 'test': None} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None, 'test': None} for name in self.target_names}
        
        for name in self.target_names:
            if 'lfw' in name.lower():
                lfw_path = os.path.join(root, 'LFW')
                self.lfw_dataset = LFW(lfw_path, os.path.join(lfw_path, 'pairs.txt'), landmark_file_path=landmarks_path, transform=transform_test)
            else:
                dataset = init_imgreid_dataset(
                    root=self.root, name=name, target='val', split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                    cuhk03_classic_split=self.cuhk03_classic_split, **kwargs
                )

                self.testdataset_dict[name]['query'] = dataset.query
                self.testdataset_dict[name]['gallery'] = dataset.gallery
                self.testdataset_dict[name]['test'] = dataset.query + dataset.gallery

                self.testloader_dict[name]['query'] = test_dataloader(dataset.query)
                self.testloader_dict[name]['gallery'] = test_dataloader(dataset.gallery)

                self.testloader_dict[name]['test'] = DataLoader(
                    ImageDataset(self.testdataset_dict[name]['test'], transform=transform_test),
                    sampler=sampler(self.testdataset_dict[name]['test']),
                    batch_size=self.train_batch_size, shuffle=shuffle, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=True
                )


        print("\n")
        print("  **************** Summary ****************")
        print("  train names      : {}".format(self.source_names))
        print("  # train datasets : {}".format(len(self.source_names)))
        print("  # train ids      : {}".format(self._num_train_pids))
        print("  # train images   : {}".format(len(self.train)))
        print("  # train cameras  : {}".format(self._num_train_cams))
        print("  test names       : {}".format(self.target_names))
        print("  *****************************************")
        print("\n")


class VideoDataManager(BaseDataManager):
    """
    Video-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root,
                 split_id=0,
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 seq_len=15,
                 sample_method='evenly',
                 image_training=True # train the video-reid model with images rather than tracklets
                 ):
        super(VideoDataManager, self).__init__()
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.seq_len = seq_len
        self.sample_method = sample_method
        self.image_training = image_training
        self.pin_memory = True if self.use_gpu else False

        # Build train and test transform functions
        transform_train = build_transforms(self.height, self.width, is_train=True)
        transform_test = build_transforms(self.height, self.width,is_train=False)

        print("=> Initializing TRAIN (source) datasets")
        self.train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id)

            for img_paths, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                if self.image_training:
                    # decompose tracklets into images
                    for img_path in img_paths:
                        self.train.append((img_path, pid, camid))
                else:
                    self.train.append((img_paths, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        if self.image_training:
            # each batch has image data of shape (batch, channel, height, width)
            self.trainloader = DataLoader(
                ImageDataset(self.train, transform=transform_train),
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=True
            )
        else:
            # each batch has image data of shape (batch, seq_len, channel, height, width)
            # note: this requires new training scripts
            self.trainloader = DataLoader(
                VideoDataset(self.train, seq_len=self.seq_len, sample_method=self.sample_method, transform=transform_test),
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=True
            )

        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}

        for name in self.target_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id)

            self.testloader_dict[name]['query'] = DataLoader(
                VideoDataset(dataset.query, seq_len=self.seq_len, sample_method=self.sample_method, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False,
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                VideoDataset(dataset.gallery, seq_len=self.seq_len, sample_method=self.sample_method, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False,
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print("\n")
        print("  **************** Summary ****************")
        print("  train names       : {}".format(self.source_names))
        print("  # train datasets  : {}".format(len(self.source_names)))
        print("  # train ids       : {}".format(self._num_train_pids))
        if self.image_training:
            print("  # train images   : {}".format(len(self.train)))
        else:
            print("  # train tracklets: {}".format(len(self.train)))
        print("  # train cameras   : {}".format(self._num_train_cams))
        print("  test names        : {}".format(self.target_names))
        print("  *****************************************")
        print("\n")
