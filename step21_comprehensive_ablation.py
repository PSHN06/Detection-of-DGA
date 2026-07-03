import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from step8_train_lstm import LSTMClassifier, get_sld

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
    
    # 1. Load Data
    print("Loading Dataset B and preprocessed feature matrices...")
    b_df = pd.read_csv("dataset_b_external.csv")
    y_b = pd.read_csv("y_dataset_b.csv").values.ravel()
    
    X_b = pd.read_csv("X_dataset_b.csv")
    X_b_hybrid = pd.read_csv("X_dataset_b_hybrid.csv")
    
    # 2. Component A Run (Classical Features Only)
    print("\n--- Component A Run (Classical Features Only) ---")
    xgb_base = XGBClassifier()
    xgb_base.load_model("advanced_xgb_model.json")
    
    start_time = time.perf_counter()
    preds_a = xgb_base.predict(X_b)
    batch_time_a = (time.perf_counter() - start_time) * 1000  # in ms
    
    acc_a = accuracy_score(y_b, preds_a)
    cm_a = confusion_matrix(y_b, preds_a)
    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
    fnr_a = fn_a / (fn_a + tp_a) if (fn_a + tp_a) > 0 else 0.0
    
    print(f"Component A Accuracy: {acc_a*100:.4f}%")
    print(f"Component A FNR:      {fnr_a*100:.4f}%")
    print(f"Component A Batch Time: {batch_time_a:.2f} ms")
    
    # 3. Component B Run (Sequence Features Only)
    print("\n--- Component B Run (Sequence Features Only) ---")
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    lstm_opt = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_opt.load_state_dict(torch.load("optimized_window_lstm.pt", map_location=device, weights_only=False))
    lstm_opt.to(device)
    lstm_opt.eval()
    
    start_time = time.perf_counter()
    b_df['sld'] = b_df['domain'].apply(get_sld)
    max_len = 32
    
    # Tokenize
    sequences = []
    for domain in b_df['sld']:
        domain_str = str(domain)
        seq = [char_to_idx.get(c, 1) for c in domain_str]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        sequences.append(seq)
    seqs_arr = np.array(sequences)
    
    # Batch Sequence Inference
    probs_b = get_predictions(lstm_opt, seqs_arr, device)
    preds_b = (probs_b >= 0.5).astype(int)
    batch_time_b = (time.perf_counter() - start_time) * 1000  # in ms
    
    acc_b = accuracy_score(y_b, preds_b)
    cm_b = confusion_matrix(y_b, preds_b)
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    fnr_b = fn_b / (fn_b + tp_b) if (fn_b + tp_b) > 0 else 0.0
    
    print(f"Component B Accuracy: {acc_b*100:.4f}%")
    print(f"Component B FNR:      {fnr_b*100:.4f}%")
    print(f"Component B Batch Time: {batch_time_b:.2f} ms")
    
    # 4. Full System Run (Hybrid Stacking Ensemble)
    print("\n--- Full System Run (Hybrid Stacking Ensemble) ---")
    hybrid_xgb = XGBClassifier()
    hybrid_xgb.load_model("hybrid_xgb_model.json")
    
    start_time = time.perf_counter()
    preds_hybrid = hybrid_xgb.predict(X_b_hybrid)
    batch_time_hybrid = (time.perf_counter() - start_time) * 1000  # in ms
    
    acc_hybrid = accuracy_score(y_b, preds_hybrid)
    cm_hybrid = confusion_matrix(y_b, preds_hybrid)
    tn_hybrid, fp_hybrid, fn_hybrid, tp_hybrid = cm_hybrid.ravel()
    fnr_hybrid = fn_hybrid / (fn_hybrid + tp_hybrid) if (fn_hybrid + tp_hybrid) > 0 else 0.0
    
    print(f"Full System Accuracy: {acc_hybrid*100:.4f}%")
    print(f"Full System FNR:      {fnr_hybrid*100:.4f}%")
    print(f"Full System Batch Time: {batch_time_hybrid:.2f} ms")
    
    # 5. Generate Report Deltas
    fnr_reduction_a = fnr_a - fnr_hybrid
    acc_increase_a = acc_hybrid - acc_a
    
    fnr_reduction_b = fnr_b - fnr_hybrid
    acc_increase_b = acc_hybrid - acc_b
    
    print("\n" + "="*80)
    print("                 COMPREHENSIVE SYSTEM ABLATION STUDY REPORT                 ")
    print("="*80)
    print(f"| {'Feature Environment':<35} | {'Accuracy':<12} | {'FNR':<12} | {'Batch Time':<12} |")
    print("-"*80)
    print(f"| {'Component A (Classical Features Only)':<35} | {acc_a*100:10.4f}% | {fnr_a*100:10.4f}% | {batch_time_a:9.2f} ms |")
    print(f"| {'Component B (Sequence Features Only)':<35} | {acc_b*100:10.4f}% | {fnr_b*100:10.4f}% | {batch_time_b:9.2f} ms |")
    print(f"| {'Full Hybrid Stacking Ensemble':<35} | {acc_hybrid*100:10.4f}% | {fnr_hybrid*100:10.4f}% | {batch_time_hybrid:9.2f} ms |")
    print("="*80)
    print("\nSYSTEM IMPROVEMENT ANALYSIS:")
    print(f"[1] Stacking Fusion vs. Classical Baseline (XGBoost):")
    print(f"    - Accuracy Increase:      +{acc_increase_a*100:.4f}% (absolute)")
    print(f"    - False Negative Rate (FNR) Drop:  -{fnr_reduction_a*100:.4f}% (absolute)")
    print(f"    - Stacking sequence signals dropped missed DGA threats from {fnr_a*100:.2f}% to {fnr_hybrid*100:.2f}%!")
    
    print(f"\n[2] Stacking Fusion vs. Sequence-Only Baseline (LSTM):")
    print(f"    - Accuracy Increase:      +{acc_increase_b*100:.4f}% (absolute)")
    print(f"    - False Negative Rate (FNR) Drop:  -{fnr_reduction_b*100:.4f}% (absolute)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
