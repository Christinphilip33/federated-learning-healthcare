import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATA_DIR = "./data/UCI_HAR"
ZIP_FILE = "./data/uci_har.zip"

def download_and_extract():
    """Downloads and extracts the UCI HAR dataset if not already present."""
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(DATA_DIR):
        print("Downloading UCI HAR Dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall("./data")
        os.rename("./data/UCI HAR Dataset", DATA_DIR)
        print("Dataset ready.")
    else:
        print("UCI HAR Dataset already downloaded.")

def load_har_data(partition='train'):
    """Loads HAR data from raw txt files.
    Returns: X (N, 9, 128), y (N,)
    """
    prefix = os.path.join(DATA_DIR, partition)
    signals_dir = os.path.join(prefix, "Inertial Signals")
    
    # 9 sensors
    filenames = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    
    X_signals = []
    for filename in filenames:
        filepath = os.path.join(signals_dir, f"{filename}_{partition}.txt")
        # Load data, each row is 128 readings
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
        X_signals.append(df.values)
    
    # Shape: (N, 128, 9)
    X = np.stack(X_signals, axis=-1)
    
    # PyTorch wants channels first for Conv1d: (N, 9, 128)
    X = np.transpose(X, (0, 2, 1))
    
    # Load labels
    y_path = os.path.join(prefix, f"y_{partition}.txt")
    df_y = pd.read_csv(y_path, delim_whitespace=True, header=None)
    # Original labels are 1-6. Shift to 0-5 for PyTorch CrossEntropy
    y = df_y.values.squeeze() - 1
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_har_datasets():
    """Load both train and test as TensorDatasets."""
    download_and_extract()
    X_train, y_train = load_har_data('train')
    X_test, y_test = load_har_data('test')
    
    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)
    return trainset, testset
