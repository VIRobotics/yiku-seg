"""This package include all things related to data.

Image processing, load dataset, make dataloader etc....
"""
import os.path
from pathlib import Path

import torch.utils.data as data
from os.path import abspath, dirname

from importlib.util import spec_from_file_location
from importlib.util import module_from_spec

import torch

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, dataset):
        """Create a dataset instance given the name [dataset_mode] and a multi-threaded data loader."""
        self.opt = opt
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            num_workers=int(opt.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # if i * self.opt.batch_size >= self.opt.max_dataset_size:
            #     break
            yield data

def load_dataset(dataset_name):
    """Loading a dataset using dataset_name
    the dataset name's format is [dataset_name]_dataset.py
    """

    datasetlib_filename = os.path.join(dirname(abspath(__file__)),f"{dataset_name}_dataset.py")
    spec = spec_from_file_location(Path(datasetlib_filename).name, datasetlib_filename)
    datasetlib = module_from_spec(spec)
    spec.loader.exec_module(datasetlib)
    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, data.Dataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("Dataset is not existed ")
    return dataset


def create_dataset(opt):
    """Create dataset with dataloader."""
    dataset_def = load_dataset(opt.dataset)
    dataset = dataset_def(opt)
    data_loader = CustomDatasetDataLoader(opt, dataset)
    print("dataset [%s] was created" % type(dataset).__name__)
    dataset = data_loader.load_data()
    return dataset