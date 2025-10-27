import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512


class SimpleNN(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, device=None):
        super().__init__()
        if input_dim is not None: # allow basic initialization
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

            self.to(device)
    
    def forward(self, x):
        return self.net(x)
class SimpleNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, output_dim, epochs=5, device=None, lr=1e-3, batch_size=512):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN(input_dim=input_dim, output_dim=output_dim, device=device).to(self.device)

    def fit(self, X, y, sample_weight=None):
        self.model.train()
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X.astype(float).values if isinstance(X, pd.DataFrame) else X.astype(float), dtype=torch.float32),
            torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we weight manually
        
        for epoch in range(self.epochs):
            for batch_idx, (inputs, targets) in enumerate(loader):
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                
                losses = criterion(outputs, targets)
                
                if sample_weight is not None:
                    # Safe slice sample_weight for current batch
                    if isinstance(sample_weight, pd.Series):
                        weights_slice = sample_weight.iloc[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size].to_numpy()
                    else:
                        weights_slice = sample_weight[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
                    
                    batch_weights = torch.tensor(weights_slice, dtype=torch.float32).to(self.device)
                    losses = losses * batch_weights
                
                loss = losses.mean()
                loss.backward()
                optimizer.step()


    def predict(self, X):
        self.model.eval()
        
        if isinstance(X, torch.Tensor):
            X_tensor = X
        else:
            X_tensor = torch.tensor(X.astype(float).values if isinstance(X, pd.DataFrame) else X.astype(float), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits)
            return (probs >= 0.5).long().cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, torch.Tensor):
            X_tensor = X
        else:
            X_tensor = torch.tensor(X.astype(float).values if isinstance(X, pd.DataFrame) else X.astype(float), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            # Fairlearn expects a 2D array of shape (n_samples, 2)
            return np.stack([1 - probs, probs], axis=1)

    def set_device(self, device):
        self.model.to(device)