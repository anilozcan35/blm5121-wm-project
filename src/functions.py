import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset_from_kaggle(user_name: str, dataset_name: str, path: str):
    if not os.path.isfile(os.path.join(path, "bodyPerformance.csv")):
        os.makedirs(path, exist_ok=True)
        # Kaggle API objesini oluşturun
        api = KaggleApi()

        # API anahtarını yükleyin
        api.authenticate()

        # Veri setini indirin
        api.dataset_download_files(user_name + '/' + dataset_name, path=path, unzip=True)
    else:
        print("Dataset already exists!")