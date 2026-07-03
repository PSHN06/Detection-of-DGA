import os
import time
import json
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, dropout_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lengths = torch.clamp((x != 0).sum(dim=1), min=1).cpu()
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_lstm_out, _ = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out, batch_first=True, total_length=x.size(1)
        )
        mask = (x == 0).unsqueeze(-1)
        masked_lstm_out = lstm_out.masked_fill(mask, -1e9)
        last_out, _ = torch.max(masked_lstm_out, dim=1)
        out = self.dropout(last_out)
        out = self.fc(out)
        return self.sigmoid(out)

def get_sld(d):
    d_str = str(d).lower().strip()
    if d_str.startswith('www.'):
        d_str = d_str[4:]
    return d_str.split('.')[0]

def tokenize_and_pad(domains, char_to_idx, max_len=64):
    sequences = []
    for domain in domains:
        domain_str = str(domain)
        seq = [char_to_idx.get(c, 1) for c in domain_str]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        sequences.append(seq)
    return np.array(sequences)

def predict_lstm(model, tokenized_seqs, batch_size=512, device='cpu'):
    preds = []
    num_samples = len(tokenized_seqs)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = tokenized_seqs[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.long).to(device)
            outputs = model(batch_tensor).squeeze(1)
            probs = outputs.cpu().numpy()
            batch_preds = (probs >= 0.5).astype(int)
            preds.extend(batch_preds)
    return np.array(preds)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return acc, f1, fpr, fnr

def print_table(title, results):
    print("="*80)
    print(f" {title.center(78)} ")
    print("="*80)
    print(f"| {'Model Name':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'FPR':<10} | {'FNR':<10} |")
    print("-"*80)
    for model_name, acc, f1, fpr, fnr in results:
        print(f"| {model_name:<25} | {acc:<10.6f} | {f1:<10.6f} | {fpr:<10.6f} | {fnr:<10.6f} |")
    print("="*80 + "\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load preprocessed Test Partition datasets
    print("Loading hold-out test datasets...")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()
    raw_test = pd.read_csv("test.csv")
    
    # 2. Load preprocessed Dataset B External Partition datasets
    print("Loading blind Dataset B datasets...")
    X_b = pd.read_csv("X_dataset_b.csv")
    y_b = pd.read_csv("y_dataset_b.csv").values.ravel()
    raw_b = pd.read_csv("dataset_b_external.csv")
    
    # 3. Load Models
    print("Loading saved model files...")
    # LightGBM
    lgb_model = joblib.load("baseline_lgb_model.pkl")
    # XGBoost
    xgb_model = XGBClassifier()
    xgb_model.load_model("advanced_xgb_model.json")
    # LSTM vocab and model weights
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    lstm_model = LSTMClassifier(vocab_size=vocab_size)
    lstm_model.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device))
    lstm_model.to(device)
    lstm_model.eval()
    
    # 4. Process LSTM domains
    print("Tokenizing test and external validation domains...")
    test_seqs = tokenize_and_pad(raw_test['domain'].apply(get_sld), char_to_idx)
    b_seqs = tokenize_and_pad(raw_b['domain'].apply(get_sld), char_to_idx)
    
    # 5. Evaluate on Hold-out Test Partition
    print("Running inferences on Test Partition...")
    # LGB
    lgb_preds_test = lgb_model.predict(X_test)
    lgb_metrics_test = compute_metrics(y_test, lgb_preds_test)
    
    # XGB
    xgb_preds_test = xgb_model.predict(X_test)
    xgb_metrics_test = compute_metrics(y_test, xgb_preds_test)
    
    # LSTM
    lstm_preds_test = predict_lstm(lstm_model, test_seqs, device=device)
    lstm_metrics_test = compute_metrics(y_test, lstm_preds_test)
    
    test_results = [
        ("LightGBM (Baseline)", *lgb_metrics_test),
        ("XGBoost (Advanced)", *xgb_metrics_test),
        ("LSTM (Sequence Deep)", *lstm_metrics_test)
    ]
    
    # 6. Evaluate on Blind Dataset B External Partition
    print("Running inferences on Dataset B External Partition...")
    # LGB
    lgb_preds_b = lgb_model.predict(X_b)
    lgb_metrics_b = compute_metrics(y_b, lgb_preds_b)
    
    # XGB
    xgb_preds_b = xgb_model.predict(X_b)
    xgb_metrics_b = compute_metrics(y_b, xgb_preds_b)
    
    # LSTM
    lstm_preds_b = predict_lstm(lstm_model, b_seqs, device=device)
    lstm_metrics_b = compute_metrics(y_b, lstm_preds_b)
    
    b_results = [
        ("LightGBM (Baseline)", *lgb_metrics_b),
        ("XGBoost (Advanced)", *xgb_metrics_b),
        ("LSTM (Sequence Deep)", *lstm_metrics_b)
    ]
    
    # 7. Print Comparative Tables
    print_table("TEST PARTITION COMPARATIVE EVALUATION (DATASET A)", test_results)
    print_table("EXTERNAL GENERALIZATION STUDY (DATASET B - UNSEEN FAMILIES/TIMELINE)", b_results)

if __name__ == "__main__":
    main()
