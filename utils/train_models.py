import pandas as pd
import torch
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split
from fairlearn.preprocessing import CorrelationRemover
from copy import deepcopy
from joblib import Parallel, delayed

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_model import SimpleNN, device, BATCH_SIZE
from helper_funcs import load_datasets, initialize_models

print(f"Using device: {device}")

def train_model(args):
    model, X, y = args
    model.fit(X, y)
    return model


### 1 - Load Data + Models
datasets = load_datasets()
for (data, name) in datasets:
    print()
    print(f"Processing {name}...")
    X = data['data'].iloc[:, :]
    y = data['target']#[:BATCH_SIZE+1000]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # fairness preprocessing
    cols = list(X_train.columns)
    cols.remove("race")
    cr = CorrelationRemover(sensitive_feature_ids=['race'], alpha=0.75)
    cr.fit(X_train)

    X_train_cr = pd.DataFrame(cr.transform(X_train), columns=cols)
    X_train_cr['race'] = X_train['race'].values
    X_train_cr = X_train_cr[X_train.columns]
    X_test_cr = pd.DataFrame(cr.transform(X_test), columns=cols)
    X_test_cr['race'] = X_test['race'].values
    X_test_cr = X_test_cr[X_test.columns]

    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    X_train_cr, X_test_cr = X_train_cr.reset_index(drop=True), X_test_cr.reset_index(drop=True)

    # initialize models
    nn_params = (X_train.shape[1], 1, device)
    models = initialize_models(nn_params)
    models_cr = deepcopy(models)

### 2 -- Train models
    print("Training Original Models...")
    trained = process_map(train_model, [(model, X_train, y_train) for model_name, model in models.items() if not isinstance(model,SimpleNN)], max_workers=4, chunksize=1)
    models['Neural Network'].fit(X_train, y_train)
    trained.append(models['Neural Network'])
    print("Training Fair Models...")
    trained_cr = process_map(train_model, [(model, X_train_cr, y_train) for model_name, model in models.items() if not isinstance(model,SimpleNN)], max_workers=4, chunksize=1)
    models_cr['Neural Network'].fit(X_train_cr, y_train)
    trained_cr.append(models_cr['Neural Network'])    
        
    for (model_name, model), (model_name_cr, model_cr) in zip(models.items(), models_cr.items()):
        if isinstance(model, SimpleNN):
            torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'NN_{name}.pth'))
            torch.save(model_cr.state_dict(), os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'NN_{name}_cr.pth'))
        else:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'{model_name}_{name}.pkl'), 'wb') as f:
                pkl.dump(model, f)
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'{model_name_cr}_{name}_cr.pkl'), 'wb') as f:
                pkl.dump(model_cr, f)
