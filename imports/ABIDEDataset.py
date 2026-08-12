import torch
from torch_geometric.data import InMemoryDataset,Data
import os
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
import pickle
from imports.read_abide_stats_parall import read_data


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return


    def process(self):
        self.data, self.slices = read_data(self.raw_dir)  # already returns (Data, slices)
        file_names = self.raw_file_names

        # Use PyG's built-in unstacking based on slices
        data_list = [self.get(i) for i in range(self.len())]

        assert len(file_names) == len(data_list), "Mismatch between filenames and loaded data"

        for i, data in enumerate(data_list):
            patient_id = file_names[i].split('.')[0]
            data.patient_id = patient_id

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Re-collate after patient_id was added
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
