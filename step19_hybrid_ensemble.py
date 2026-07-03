import os
import time
import json
import math
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from step8_train_lstm import LSTMClassifier, get_sld

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return acc, f1, fpr, fnr

class InferenceDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = torch.tensor(seqs, dtype=torch.long)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx]

def get_predictions(model, seqs, device):
    dataset = InferenceDataset(seqs)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs.extend(out.squeeze(1).cpu().numpy())
    return np.array(probs)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for sequence inference: {device}")
    
    # 1. Load Vocab
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # 2. Load Optimized Window LSTM Model
    print("Loading optimized window LSTM model (hidden_dim=32, max_len=32)...")
    lstm_opt = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_opt.load_state_dict(torch.load("optimized_window_lstm.pt", map_location=device, weights_only=False))
    lstm_opt.to(device)
    lstm_opt.eval()
    
    # Tokenizer helper
    max_len = 32
    def tokenize_and_pad(domains):
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
        
    # 3. Load datasets and run batch inference
    print("Loading raw domain text files...")
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    b_df = pd.read_csv("dataset_b_external.csv")
    
    train_df['sld'] = train_df['domain'].apply(get_sld)
    val_df['sld'] = val_df['domain'].apply(get_sld)
    test_df['sld'] = test_df['domain'].apply(get_sld)
    b_df['sld'] = b_df['domain'].apply(get_sld)
    
    print("Tokenizing SLDs...")
    seqs_train = tokenize_and_pad(train_df['sld'])
    seqs_val = tokenize_and_pad(val_df['sld'])
    seqs_test = tokenize_and_pad(test_df['sld'])
    seqs_b = tokenize_and_pad(b_df['sld'])
    
    print("Running batch sequence inference to generate lstm_prob features...")
    lstm_prob_train = get_predictions(lstm_opt, seqs_train, device)
    lstm_prob_val = get_predictions(lstm_opt, seqs_val, device)
    lstm_prob_test = get_predictions(lstm_opt, seqs_test, device)
    lstm_prob_b = get_predictions(lstm_opt, seqs_b, device)
    
    # 4. Load scaled feature sets and concatenate
    print("Loading scaled classical features...")
    X_train = pd.read_csv("X_train.csv")
    X_val = pd.read_csv("X_val.csv")
    X_test = pd.read_csv("X_test.csv")
    X_b = pd.read_csv("X_dataset_b.csv")
    
    y_train = pd.read_csv("y_train.csv").values.ravel()
    y_val = pd.read_csv("y_val.csv").values.ravel()
    y_test = pd.read_csv("y_test.csv").values.ravel()
    y_b = pd.read_csv("y_dataset_b.csv").values.ravel()
    
    print("Augmenting feature matrices with lstm_prob...")
    X_train['lstm_prob'] = lstm_prob_train
    X_val['lstm_prob'] = lstm_prob_val
    X_test['lstm_prob'] = lstm_prob_test
    X_b['lstm_prob'] = lstm_prob_b
    
    # Save augmented matrices
    print("Saving augmented feature matrices...")
    X_train.to_csv("X_train_hybrid.csv", index=False)
    X_val.to_csv("X_val_hybrid.csv", index=False)
    X_test.to_csv("X_test_hybrid.csv", index=False)
    X_b.to_csv("X_dataset_b_hybrid.csv", index=False)
    
    # 5. Train Hybrid Stacking Models
    print("Training Hybrid Stacking LightGBM meta-classifier...")
    start = time.time()
    hybrid_lgb = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    hybrid_lgb.fit(X_train, y_train)
    lgb_train_time = time.time() - start
    print(f"LightGBM training completed in {lgb_train_time:.2f} seconds.")
    joblib.dump(hybrid_lgb, "hybrid_lgb_model.pkl")
    
    print("Training Hybrid Stacking XGBoost meta-classifier...")
    start = time.time()
    hybrid_xgb = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    hybrid_xgb.fit(X_train, y_train)
    xgb_train_time = time.time() - start
    print(f"XGBoost training completed in {xgb_train_time:.2f} seconds.")
    hybrid_xgb.save_model("hybrid_xgb_model.json")
    
    # 6. Evaluation & Generalization Benchmarking
    print("\nRunning baseline and hybrid evaluations on Hold-out Test Partition (Dataset A)...")
    # Baseline models for reference
    base_lgb = joblib.load("baseline_lgb_model.pkl")
    base_xgb = XGBClassifier()
    base_xgb.load_model("advanced_xgb_model.json")
    
    # Base evaluations on Test
    X_test_base = X_test.drop(columns=['lstm_prob'])
    lgb_preds_base_test = base_lgb.predict(X_test_base)
    xgb_preds_base_test = base_xgb.predict(X_test_base)
    
    # Hybrid evaluations on Test
    lgb_preds_hybrid_test = hybrid_lgb.predict(X_test)
    xgb_preds_hybrid_test = hybrid_xgb.predict(X_test)
    
    metrics_lgb_base_test = compute_metrics(y_test, lgb_preds_base_test)
    metrics_xgb_base_test = compute_metrics(y_test, xgb_preds_base_test)
    metrics_lgb_hybrid_test = compute_metrics(y_test, lgb_preds_hybrid_test)
    metrics_xgb_hybrid_test = compute_metrics(y_test, xgb_preds_hybrid_test)
    
    # Base evaluations on Dataset B
    print("Running baseline and hybrid evaluations on External Generalization Study (Dataset B)...")
    X_b_base = X_b.drop(columns=['lstm_prob'])
    lgb_preds_base_b = base_lgb.predict(X_b_base)
    xgb_preds_base_b = base_xgb.predict(X_b_base)
    
    # Hybrid evaluations on Dataset B
    lgb_preds_hybrid_b = hybrid_lgb.predict(X_b)
    xgb_preds_hybrid_b = hybrid_xgb.predict(X_b)
    
    metrics_lgb_base_b = compute_metrics(y_b, lgb_preds_base_b)
    metrics_xgb_base_b = compute_metrics(y_b, xgb_preds_base_b)
    metrics_lgb_hybrid_b = compute_metrics(y_b, lgb_preds_hybrid_b)
    metrics_xgb_hybrid_b = compute_metrics(y_b, xgb_preds_hybrid_b)
    
    # 7. Print Comparative Table
    print("\n" + "="*95)
    print("                     HYBRID STACKING ENSEMBLE GENERALIZATION STUDY                     ")
    print("="*95)
    print(f"| {'Model Type & Name':<28} | {'Dataset':<18} | {'Accuracy':<10} | {'F1-Score':<10} | {'FPR':<8} | {'FNR':<8} |")
    print("-"*95)
    
    # Test set results
    print(f"| {'Baseline LightGBM':<28} | {'Test (Dataset A)':<18} | {metrics_lgb_base_test[0]:<10.6f} | {metrics_lgb_base_test[1]:<10.6f} | {metrics_lgb_base_test[2]:<8.6f} | {metrics_lgb_base_test[3]:<8.6f} |")
    print(f"| {'Hybrid Stacking LGB':<28} | {'Test (Dataset A)':<18} | {metrics_lgb_hybrid_test[0]:<10.6f} | {metrics_lgb_hybrid_test[1]:<10.6f} | {metrics_lgb_hybrid_test[2]:<8.6f} | {metrics_lgb_hybrid_test[3]:<8.6f} |")
    print("-"*95)
    print(f"| {'Baseline XGBoost':<28} | {'Test (Dataset A)':<18} | {metrics_xgb_base_test[0]:<10.6f} | {metrics_xgb_base_test[1]:<10.6f} | {metrics_xgb_base_test[2]:<8.6f} | {metrics_xgb_base_test[3]:<8.6f} |")
    print(f"| {'Hybrid Stacking XGB':<28} | {'Test (Dataset A)':<18} | {metrics_xgb_hybrid_test[0]:<10.6f} | {metrics_xgb_hybrid_test[1]:<10.6f} | {metrics_xgb_hybrid_test[2]:<8.6f} | {metrics_xgb_hybrid_test[3]:<8.6f} |")
    print("="*95)
    
    # Dataset B results
    print(f"| {'Baseline LightGBM':<28} | {'External (Dataset B)':<18} | {metrics_lgb_base_b[0]:<10.6f} | {metrics_lgb_base_b[1]:<10.6f} | {metrics_lgb_base_b[2]:<8.6f} | {metrics_lgb_base_b[3]:<8.6f} |")
    print(f"| {'Hybrid Stacking LGB':<28} | {'External (Dataset B)':<18} | {metrics_lgb_hybrid_b[0]:<10.6f} | {metrics_lgb_hybrid_b[1]:<10.6f} | {metrics_lgb_hybrid_b[2]:<8.6f} | {metrics_lgb_hybrid_b[3]:<8.6f} |")
    print("-"*95)
    print(f"| {'Baseline XGBoost':<28} | {'External (Dataset B)':<18} | {metrics_xgb_base_b[0]:<10.6f} | {metrics_xgb_base_b[1]:<10.6f} | {metrics_xgb_base_b[2]:<8.6f} | {metrics_xgb_base_b[3]:<8.6f} |")
    print(f"| {'Hybrid Stacking XGB':<28} | {'External (Dataset B)':<18} | {metrics_xgb_hybrid_b[0]:<10.6f} | {metrics_xgb_hybrid_b[1]:<10.6f} | {metrics_xgb_hybrid_b[2]:<8.6f} | {metrics_xgb_hybrid_b[3]:<8.6f} |")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
