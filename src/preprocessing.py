import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import get_list_of_img_paths, make_class_to_idx_map
from os.path import dirname
from glob import glob

def download_food101():
    download_path = '/*/*/.cache/kagglehub/datasets/dansbecker/food-101/versions/1'
    existing_paths = glob(download_path)
    if len(existing_paths) == 1:
        return existing_paths[0]
    else:
        kagglehub.login()
        dataset_path = kagglehub.dataset_download('dansbecker/food-101')
        return dataset_path

def build_image_path_label_df(data_root, is_for_test = False):
    if not is_for_test:
        data_root = data_root + '/food-101/food-101'

    img_paths = get_list_of_img_paths(data_root, 'images')
    map = make_class_to_idx_map(data_root, 'images')
    class_names = [ dirname(path.replace('images/','')) for path in img_paths ]
    labels = [ map[class_name] for class_name in class_names ]
    data = {'path': img_paths, 'label': labels}

    if is_for_test:
        return class_names, labels, pd.DataFrame(data)

    return pd.DataFrame(data)

def split_dataset(dataset_df, train_size = 0.6, val_size = 0.1, test_size = 0.3, csv_path = None):
    if len(dataset_df) == 0:
        raise ValueError('Dataset is empty.')

    train_df, test_val_df = train_test_split(dataset_df, test_size = val_size + test_size)
    if val_size:
        val_df, test_df = train_test_split(test_val_df, test_size = test_size/(test_size + val_size))

    if csv_path is not None:
        os.makedirs(csv_path, exist_ok=True)
        train_df.to_csv(csv_path + '/train.csv', index = None)
        test_df.to_csv(csv_path + '/test.csv', index = None)
        if val_size:
            val_df.to_csv(csv_path + '/val.csv', index = None)

    if val_size:
        return train_df, val_df, test_df
    else:
        return train_df, test_val_df

