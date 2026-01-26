import kagglehub


def download_food101():
    kagglehub.login()
    dataset_path = kagglehub.dataset_download('dansbecker/food-101')
    return dataset_path