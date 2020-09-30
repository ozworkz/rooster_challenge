import os
from utils.download import download
from utils.unzip import unzip
from utils.prep_dataset import generate_rooster_dataset
from utils.rooster_utils import load_configurations

#Dataset for Environmental Sound Classification
ECS_DATA_SET_URL = 'https://github.com/karoldvl/ESC-50/archive/master.zip'

DATASET_DIR = 'dataset'
ECS_DATA_DIR = os.path.join(DATASET_DIR, 'ECS-50-master')

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

if not os.path.exists(os.path.join(DATASET_DIR,ECS_DATA_SET_URL.split('/')[-1])):
    filepath = download(ECS_DATA_SET_URL,DATASET_DIR)
    print("Unzipping {}...".format(filepath))
    unzip(os.path.join(filepath), DATASET_DIR)

settings = load_configurations(mode="dataset_generator")
generate_rooster_dataset(settings)


