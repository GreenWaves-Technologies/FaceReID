import os
from .bases import BaseImageDataset


class ReIdImageFolder(BaseImageDataset):
    def __init__(self, root='data', verbose=True, clear_val=False, target='train', **kwargs):
        super(ReIdImageFolder, self).__init__()

        if target == 'train':
            train, _ = self.make_dataset(os.path.join(root, 'train'))
        else:
            train = []

        val = 'val'
        if clear_val:
            val = 'clear_val'
        if 'val' in target:
            gallery, query = self.make_dataset(os.path.join(root, val), val=True, **kwargs)
        else:
            gallery = []
            query = []

        if verbose:
            print("=> ImageFolder dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def make_dataset(self, dir, val=False, num_test_ids=0, gallery_items=20, **kwargs):
        dataset = []
        query = []
        dir = os.path.expanduser(dir)
        pid = 0
        for target_idx, target in enumerate(sorted(os.listdir(dir))):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            start = 0
            if val and len(os.listdir(d)) > 1:
                start = 1
                query.append((os.path.join(d, sorted(os.listdir(d))[0]), pid, len(os.listdir(d))))

            for frame_id, fname in enumerate(sorted(os.listdir(d))[start:]):
                if val and frame_id >= gallery_items:
                    break
                path = os.path.join(d, fname)
                item = (path, pid, frame_id + start)
                dataset.append(item)
            pid += 1
            if num_test_ids != 0 and pid == num_test_ids:
                break

        return dataset, query


class VGGFace2(ReIdImageFolder):
    def __init__(self, root='data', verbose=True, clear_val=False, target='train', **kwargs):
        dataset_dir = 'vggface2'
        super(VGGFace2, self).__init__(root=os.path.join(root, dataset_dir), verbose=verbose, clear_val=clear_val,
                                       target=target, **kwargs)


class MSCeleb1M(ReIdImageFolder):
    def __init__(self, root='data', verbose=True, clear_val=False, target='train', **kwargs):
        dataset_dir = 'msceleb1m'
        super(MSCeleb1M, self).__init__(root=os.path.join(root, dataset_dir), verbose=verbose, clear_val=clear_val,
                                        target=target, **kwargs)