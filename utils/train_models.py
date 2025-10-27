import pandas as pd
import torch
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split
from fairlearn.preprocessing import CorrelationRemover, PrototypeRepresentationLearner
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, ErrorRate
from torch_model import *
from copy import deepcopy
from joblib import Parallel, delayed

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_model import SimpleNN, device, BATCH_SIZE
from helper_funcs import load_datasets, initialize_models

print(f"Using device: {device}")

def train_model(args):
    model, X, y, fair = args
    if fair:
        # Create the fairness-aware estimator
        expgrad = ExponentiatedGradient(
            estimator=model,
            constraints=EqualizedOdds(),
            objective=ErrorRate(),
            sample_weight_name="sample_weight"
        )
        expgrad.fit(X, y, sensitive_features=X['race'])
        return expgrad
    else:
        model.fit(X, y)
    return model


### 1 - Load Data + Models
datasets = load_datasets()
for (data, name) in datasets:
    print()
    print(f"Processing {name}...")

    # X = data['data'].iloc[:BATCH_SIZE+200, :]
    # y = data['target'][:BATCH_SIZE+200]
    X = data['data']
    y = data['target']

    # print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    
    # initialize models
    nn_params = {
        'input_dim': X_train.shape[1], 
        'output_dim': 1,
        'epochs': 1000, 
        'device': device, 
        'lr': 1e-3, 
        'batch_size': 512
    }
    models = initialize_models(nn_params)
    models_cr = initialize_models(nn_params)

### 2 -- Train models
    print("Training Original Models...")
    trained = [train_model((model, X_train, y_train, 0)) for model_name, model in tqdm(models.items())]
    # models['Neural Network'].fit(X_train, y_train)
    # trained.append(models['Neural Network'])
    print("Training Fair Models...")
    trained_cr = [train_model((model, X_train, y_train, 1)) for model_name, model in tqdm(models_cr.items())]
    # models_cr['Neural Network'].fit(X_train, y_train)
    # trained_cr.append(models_cr['Neural Network'])    
        
    for i, ((model_name, untrained), (model_name_cr, untrained_cr)) in enumerate(zip(models.items(), models_cr.items())):
        model = trained[i]
        model_cr = trained_cr[i]
        if isinstance(model, SimpleNNWrapper):
            model.set_device('cpu')
            model_cr.estimator.set_device('cpu')
            
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'{model_name}_{name}.pkl'), 'wb') as f:
            pkl.dump(model, f)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'{model_name_cr}_{name}_fair.pkl'), 'wb') as f:
            pkl.dump(model_cr, f)

### 3 - save test data for explanations
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_X_train.pkl'), 'wb') as f:
        X_train.to_pickle(f)
    # with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_X_train_cr.pkl'), 'wb') as f:
    #     X_train_cr.to_pickle(f)
    # with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_X_train_cr.pkl'), 'wb') as f:
    #     pkl.dump(X_train_cr, f)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_X_test.pkl'), 'wb') as f:
        X_test.to_pickle(f)
    # with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_X_test_cr.pkl'), 'wb') as f:
    #     X_test_cr.to_pickle(f)
    # with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_X_test_cr.pkl'), 'wb') as f:
    #     pkl.dump(X_test_cr, f)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_y_train.pkl'), 'wb') as f:
        y_train.to_pickle(f)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f'{name}_y_test.pkl'), 'wb') as f:
        y_test.to_pickle(f)
    # with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'{name}_cr.pkl'), 'wb') as f:
    #     pkl.dump(cr, f)