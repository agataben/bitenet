import pandas as pd
import os
from torch.utils import data
from PIL import Image


class Food101DataSet(data.Dataset):
    def __init__(self, data_root, csv, transform = None):
        self.data_root = data_root
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_path, img_label = self.data.iloc[i]['path'], self.data.iloc[i].label

        if self.transform is not None:
            img = Image.open(os.path.join(self.data_root, img_path)).convert('RGB')
            img = self.transform(img)

        return img, img_label

