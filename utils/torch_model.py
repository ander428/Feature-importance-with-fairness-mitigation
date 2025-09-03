import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
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
            nn.Linear(64, output_dim)
        )

        self.to(device)
    
    def forward(self, x):
        return self.net(x)
    
    def _epoch(self, train_loader, optimizer, criterion, device):
        self.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            preds = self(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)
    
    def fit(self, X, y, epochs=1000, lr=1e-3, report=0):
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        train_ds = SimpleDataset(X, y)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(epochs), desc="Training"):
            train_loss = self._epoch(train_loader, optimizer, criterion, device)

        if report:
            return self._evaluate(train_loader, criterion, device)


    def _evaluate(self, loader, criterion, device):
        self.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device).unsqueeze(1).float()
                preds = self(X)

                loss = criterion(preds, y)
                total_loss += loss.item()
                
                probs = torch.sigmoid(preds)
                binary_preds = (probs > 0.5).long()

                all_preds.extend(binary_preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        return acc
