import torch
from torch.utils import data
from os.path import join
from PIL import Image
import pandas as pd

class Food101DataSet(data.Dataset):
    def __init__(self, data_root, csv, transform=None):
        self.data_root = data_root
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        im_path, im_label = self.data.iloc[i]['path'], self.data.iloc[i].label

        if self.transform is not None:
            im = self.transform(im)

        return im, im_label

