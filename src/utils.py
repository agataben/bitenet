import torch
import random
import numpy as np
import pandas as pd
import os
import yaml
import csv

from os.path import basename
from os.path import dirname
from glob import glob
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# Seed setting
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_toy_dataset(data_root = 'tests', samplings_n = 3,
                      w = 384, h = 384, chanels_n = 3, csv_path = None):

    paths, labels = [], []
    for i in range(samplings_n):
        array = (np.random.rand(w, h, chanels_n) * 255).astype(np.uint8)
        img = Image.fromarray(array)
        class_name = 'class_' + f'{i}' 
        labels.append(class_name)
        os.makedirs(data_root + '/images/' + class_name, exist_ok = True)
        img_name = f'{i}' + '.jpg'
        path = data_root + '/images/' + class_name + '/' + img_name
        img.save(path)
        paths.append(path.replace(data_root + '/', ''))

    if csv_path is not None:
        path_label_dict = {'path': paths, 'labels': labels}
        path_label_df = pd.DataFrame(path_label_dict)
        path_label_df.to_csv(csv_path + '/toy_csv.csv' )

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

def calculate_mean_and_std(csv_path, data_root, yaml_path = None):
    dataset_df = pd.read_csv(csv_path)
    transform = transforms.ToTensor()

    mean = torch.zeros(3)
    mean_sq = torch.zeros(3)

    for i in range(len(dataset_df)):
        path = dataset_df.iloc[i]['path']
        sample = Image.open(os.path.join(data_root, path))
        sample = transform(sample)
        c = sample.shape[0]
        x = sample.view(c, -1)
        mean += x.mean(dim=1)
        mean_sq += (x ** 2).mean(dim=1)

    mean /= len(dataset_df)
    std = torch.sqrt(mean_sq / len(dataset_df) - mean ** 2)

    if yaml_path is not None:
        mean_list = mean.tolist()
        std_list = std.tolist()
        params = {'mean': mean_list,
                  'std': std_list}
        os.makedirs(yaml_path, exist_ok = True)
        with open(os.path.join(yaml_path, 'normalizzation.yaml'), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

    return mean, std

def get_norm_parameters(yaml_path):
    with open(os.path.join(yaml_path, 'normalizzation.yaml'), 'r') as f:
        try:
            params = yaml.safe_load(f)
        except exception as e:
            raise Exception(f'Missing normalizzation.yaml file in {yaml_path}.')
    mean = params['mean']
    std = params['std']
    return torch.Tensor(mean), torch.Tensor(std)

def get_list_of_img_paths(img_dir_path, img_dir_name):
    imgs_common_path = img_dir_path + '/' + img_dir_name + '/*/*'
    img_paths = glob(imgs_common_path)
    img_paths_from_img_dir = [ path.replace(img_dir_path + '/','') for path in img_paths ]

    return img_paths_from_img_dir

def make_class_to_idx_map(img_dir_path, img_dir_name, cvs_path = None):

    classes_common_path = img_dir_path + '/' + img_dir_name + '/*'
    class_names = [ basename(path) for path in glob(classes_common_path) ]
    class_names.sort()
    indexes = [i for i in range(len(class_names))]
    class_to_idx_dict = {class_name: idx for class_name, idx in zip(class_names, indexes)}
    if cvs_path is not None:
        class_to_idx_dict_df = pd.DataFrame({'class': class_names, 'index': indexes})
        class_to_idx_dict_df.to_csv(cvs_path + '/classes.csv')

    return class_to_idx_dict

def get_class_index_from_path(image_path, class_to_idx_dict):
    class_path = dirname(image_path)
    class_name = basename(class_path)
    return class_to_idx_dict[class_name]

def transform_input_img(img_path, path_to_norm_params_file):
    if not os.path.exists(img_path):
        raise ValueError(f'Path {img_path} does not exist')
    img = Image.open(img_path).convert('RGB')

    mean, std = get_norm_parameters(path_to_norm_params_file)
    input_transf = transforms.Compose([ transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std),
                                        transforms.Lambda(lambda x: x.unsqueeze(0))
                                        ])
    return input_transf(img)

def plot_dataset(dataset_path, class_to_indx_csv):
    if not os.path.exists(dataset_path):
        raise ValueError(f'Path {dataset_path} does not exist.')
    if not os.path.exists(class_to_indx_csv):
        raise ValueError(f'Path {class_to_indx_csv} does not exist.')

    with open(class_to_indx_csv) as f:
        class_to_indx_dict = csv.DictReader(f)

    img_paths = glob(os.path.join(dataset_path, '*.jpg'))
    if len(img_paths) == 0:
        raise ValueError(f'No images in the selected path.')

    plt.figure(figsize = (18, 6))
    for i, img_path in enumerate(img_paths):
        if i >= 27:
            break
        food_class = basename(img_path).replace('.jpg', '')
        plt.subplot(3, 9, i + 1)
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)
        plt.title(f'{food_class} ({class_to_indx_dict[food_class]})')
        plt.axis('off')

    plt.show()

def plot_dataset_composition(cvs_path, is_for_test = False):
    if not os.path.exists(cvs_path):
        raise ValueError(f'Path {cvs_path} does not exist.')

    counts = {}
    with open(cvs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row['path']
            food = dirname(path.replace('images/',''))
            if food not in counts.keys():
                counts[food] = 0
            counts[food]+= 1

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize = (12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation = 45, ha = 'right')
    plt.show()
    if is_for_test:
        return counts

