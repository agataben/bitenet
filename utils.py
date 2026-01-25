import torch
import random
import numpy as np
import pandas as pd

from os.path import basename
from glob import glob


# Seed setting
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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

def make_class_to_idx_map(img_dir_path, img_dir_name, cvs_main_path, write_csv = False):

    classes_common_path = img_dir_path + '/' + img_dir_name + '/*'
    class_names = [ basename(path) for path in glob(classes_common_path) ]
    class_to_idx_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

    if write_csv:
        class_to_idx_dict_df = pd.DataFrame(class_to_idx_dict)
        class_to_idx_dict_df.to_csv(cvs_main_path + '/classes.csv')

    return class_to_idx_dict

def get_class_index_from_path(image_path, class_to_idx_dict):
    class_name = basename(image_path)
    return class_to_idx_dict[class_name]

