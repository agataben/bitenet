import kagglehub
import os
import pandas as pd
from utils import get_list_of_img_paths, make_class_to_idx_map
from os.path import dirname


def download_food101():
    kagglehub.login()
    dataset_path = kagglehub.dataset_download('dansbecker/food-101')
    return dataset_path

def build_image_path_label_df(data_root, is_for_test = False):
    if not is_for_test:
        data_root = os.path.join(data_root, '/food-101/food-101')

    img_paths = get_list_of_img_paths(data_root, 'images')
    map = make_class_to_idx_map(data_root, 'images')
    class_names = [ dirname(path.replace('images/','')) for path in img_paths ]
    labels = [ map[class_name] for class_name in class_names ]
    data = {'path': img_paths, 'label': labels}

    if is_for_test:
        return class_names, labels, pd.DataFrame(data)

    return pd.Dataframe(data)

