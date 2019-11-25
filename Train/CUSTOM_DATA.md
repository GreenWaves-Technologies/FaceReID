# Training on your custom data

To train on your custom data you should:

1. Create dataset class in `torchreid/datasets`. It should look like:

```python
class DummyDataset:
    def __init__(self, root):
        self.train = []
        self.query = []
        self.gallery = []
```

`root` - path to the data source
`train`, `query` and `gallery` should contain tuples `(img_path, pid, camid)`, where

* `img_path` - path to the image
* `pid` (person identifier) - every unique object should have it's own pid.
* `camid` (camera identifier) - every view of object should have it's own camid.
`camid` is only used to remove gallery samples that have the same pid and camid with query

Note: If you want to use the dataset only for training, you need not to define query and gallery

2. Import your dataset class into `torchreid/datasets/__init__.py`.
Add your dataset to `__imgreid_factory`.

```python
...
'sensereid': SenseReID,
'dataset_name': YourDatasetClassName
}
```

3. Provide dataset name to the training script
```bash
python train_imgreid.py \
-s dataset_name \ # source dataset for training
-t dataset_name \ # target dataset for test
...
```