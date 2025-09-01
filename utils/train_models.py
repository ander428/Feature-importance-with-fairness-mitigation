import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_model import SimpleNN, device, BATCH_SIZE
from helper_funcs import load_datasets, initialize_models

print(f"Using device: {device}")

### 1 - Load Data
datasets = load_datasets()

for (data, name) in datasets:
    X = data['data'].iloc[:BATCH_SIZE, :]
    y = data['target'][:BATCH_SIZE]
    
    nn_params = (X.shape[1], 1, device)
    models = initialize_models(nn_params)

    break
