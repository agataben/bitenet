import torch
import random
import numpy as np
import pandas as pd
import os

from os.path import basename
from os.path import dirname
from glob import glob
from PIL import Image


# Seed setting
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_toy_dataset(data_root = 'tests/images', samplings_n = 3,
                      w = 384, h = 384, chanels_n = 3):

    for i in range(samplings_n):
        array = (np.random.rand(w, h, chanels_n) * 255).astype(np.uint8)
        img = Image.fromarray(array)
        class_name = 'class_' + f'{i}' 
        os.makedirs(data_root + '/' + class_name, exist_ok = True)
        img_name = f'{i}' + '.jpg'
        img.save(data_root + '/' + class_name + '/' + img_name)

class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self,value,num):
        self.sum += value*num
        self.num += num

    def value(self):
        try:
            return self.sum/self.num
        except:
            return None

def calculate_mean_and_std(x):
    canals_n = x[0][0].shape[0]

    mean = torch.zeros(canals_n)
    for sample in x:
        mean += sample[0].view(canals_n, -1).mean(dim=1)
    mean /= len(x)

    std = torch.zeros(canals_n)
    for sample in x:
        std += ((sample[0].view(canals_n, -1) - mean[:, None])**2).mean(dim=1)
    std = torch.sqrt(std / len(x))

    return mean, std

def get_list_of_img_paths(img_dir_path, img_dir_name):
    imgs_common_path = img_dir_path + '/' + img_dir_name + '/*/*'
    img_paths = glob(imgs_common_path)
    img_paths_from_img_dir = [ path.replace(img_dir_path + '/','') for path in img_paths ]

    return img_paths_from_img_dir

def make_class_to_idx_map(img_dir_path, img_dir_name, cvs_path = None):

    classes_common_path = img_dir_path + '/' + img_dir_name + '/*'
    class_names = [ basename(path) for path in glob(classes_common_path) ]
    class_names.sort()
    class_to_idx_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

    if cvs_path is not None:
        class_to_idx_dict_df = pd.DataFrame(class_to_idx_dict)
        class_to_idx_dict_df.to_csv(cvs_path + '/classes.csv')

    return class_to_idx_dict

def get_class_index_from_path(image_path, class_to_idx_dict):
    class_path = dirname(image_path)
    class_name = basename(class_path)
    return class_to_idx_dict[class_name]

