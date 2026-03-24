import os
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import TensorDataset

DATA_URL = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
DATA_DIR = "./data/WISDM"
ZIP_FILE = "./data/wisdm.zip"
PROCESSED_FILE = "./data/wisdm_processed.pt"

def download_and_extract():
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(DATA_DIR):
        print("Downloading WISDM Dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
    inner_dir = os.path.join(DATA_DIR, "wisdm-dataset")
    inner_zip = os.path.join(DATA_DIR, "wisdm-dataset.zip")
    if not os.path.exists(inner_dir) and os.path.exists(inner_zip):
        print("Extracting inner zip...")
        with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    print("Dataset ready.")

def load_wisdm_watch_accel(window_size=128, stride=64):
    if os.path.exists(PROCESSED_FILE):
        return torch.load(PROCESSED_FILE)
        
    download_and_extract()
    base_path = os.path.join(DATA_DIR, "wisdm-dataset", "raw", "watch", "accel")
    
    all_x = []
    all_y = []
    
    print("Processing raw WISDM files...")
    
    for file in os.listdir(base_path):
        if not file.endswith('.txt'):
            continue
            
        filepath = os.path.join(base_path, file)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            data = []
            labels = []
            for line in lines:
                parts = line.strip().replace(';', '').split(',')
                if len(parts) == 6:
                    activity = parts[1]
                    try:
                        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        data.append([x, y, z])
                        labels.append(activity)
                    except ValueError:
                        continue
                        
            if len(data) == 0:
                continue
                
            data = np.array(data)
            labels = np.array(labels)
            
            num_windows = (len(data) - window_size) // stride + 1
            if num_windows > 0:
                for i in range(num_windows):
                    start = i * stride
                    end = start + window_size
                    window_data = data[start:end]
                    
                    # majority vote
                    window_labels = labels[start:end]
                    unique_labels, counts = np.unique(window_labels, return_counts=True)
                    majority_label = unique_labels[np.argmax(counts)]
                    
                    all_x.append(window_data)
                    all_y.append(majority_label)
                    
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    X = np.stack(all_x)
    X = np.transpose(X, (0, 2, 1)) # (N, 3, 128)
    
    unique_labels = sorted(list(set(all_y)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in all_y])
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    print(f"Processed {len(y_tensor)} windows. Classes: {len(label_map)}")
    print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    torch.save((X_tensor, y_tensor), PROCESSED_FILE)
    
    return X_tensor, y_tensor

if __name__ == "__main__":
    X, y = load_wisdm_watch_accel()
    print("Test successful!")
