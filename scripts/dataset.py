import pandas as pd
import torch
from torch_geometric.data import Dataset, Data, HeteroData
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import random
from collections import Counter
from random import sample



class MixBench(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        raw_files = []
        return raw_files

    @property
    def processed_file_names(self):
        pt_len = len([name for name in os.listdir(osp.join(self.root, 'processed')) if os.path.isfile(osp.join(self.root, 'processed', name))])
        p = ['data_{}.pt'.format(idx) for idx in range(pt_len)]
        return p

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data