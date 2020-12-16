import os
import shutil


def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def create_directory(dir, recreate = True):
    if recreate:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    os.makedirs(dir)