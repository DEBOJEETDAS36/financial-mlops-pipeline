import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.pytorch
import mlflow.xgboost
from data_ingestion import fetch_and_prepare_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for Sequential Market Feature Extraction."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Extract last sequence state
        return self.sigmoid(out)

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int = 10):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_length],
            self.y[idx + self.seq_length]
        )

def train_pipeline(ticker: str = "AAPL"):
    df = fetch_and_prepare_data(ticker)
    
    feature_cols = ['MACD', 'RSI_14', 'Volatility_14', 'Log_Return']
    X = df[feature_cols].values
    y = df['Target_Class'].values

    # Train / Test Split (80/20 chronological)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    mlflow.set_experiment("Market_Direction_Prediction")

    # --- 1. Train XGBoost Baseline ---
    with mlflow.start_run(run_name="XGBoost_Baseline"):
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05)
        xgb_model.fit(X_train, y_train)
        preds_xgb = xgb_model.predict(X_test)
        
        acc_xgb = accuracy_score(y_test, preds_xgb)
        mlflow.log_metric("accuracy", acc_xgb)
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        print(f"XGBoost Baseline Accuracy: {acc_xgb:.4f}")

    # --- 2. Train PyTorch Bi-LSTM ---
    seq_len = 10
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length=seq_len)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_length=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BiLSTMClassifier(input_dim=len(feature_cols), hidden_dim=64).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name="PyTorch_BiLSTM"):
        model.train()
        for epoch in range(15):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        bilstm_preds, true_y = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).squeeze()
                preds = (outputs > 0.5).cpu().numpy().astype(int)
                bilstm_preds.extend(preds if preds.ndim > 0 else [preds.item()])
                true_y.extend(y_batch.numpy())

        acc_bilstm = accuracy_score(true_y, bilstm_preds)
        mlflow.log_metric("accuracy", acc_bilstm)
        mlflow.pytorch.log_model(model, "bilstm_model")
        print(f"Bi-LSTM Model Accuracy: {acc_bilstm:.4f}")

if __name__ == "__main__":
    train_pipeline("AAPL")