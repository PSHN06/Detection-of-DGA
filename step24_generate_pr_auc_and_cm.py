import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, accuracy_score
from step8_train_lstm import LSTMClassifier, get_sld
from step14_distill_gru import GRUClassifier

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
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading validation datasets...")
    val_df = pd.read_csv("val.csv")
    y_val = pd.read_csv("y_val.csv").values.ravel()
    X_val = pd.read_csv("X_val.csv")
    X_val_hybrid = pd.read_csv("X_val_hybrid.csv")
    
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    val_df['sld'] = val_df['domain'].apply(get_sld)
    
    # Pre-tokenize sequences
    def get_tokenized_arr(max_len):
        sequences = []
        for domain in val_df['sld']:
            domain_str = str(domain)
            seq = [char_to_idx.get(c, 1) for c in domain_str]
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            sequences.append(seq)
        return np.array(sequences)
        
    seqs_64 = get_tokenized_arr(64)
    seqs_32 = get_tokenized_arr(32)
    
    # 2. Load Models
    print("Loading models...")
    lgb_base = joblib.load("baseline_lgb_model.pkl")
    
    xgb_base = XGBClassifier()
    xgb_base.load_model("advanced_xgb_model.json")
    
    lstm_base = LSTMClassifier(vocab_size=vocab_size, hidden_dim=64)
    lstm_base.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device, weights_only=False))
    lstm_base.to(device)
    lstm_base.eval()
    
    gru_dist = GRUClassifier(vocab_size=vocab_size)
    gru_dist.load_state_dict(torch.load("distilled_gru.pt", map_location=device, weights_only=False))
    gru_dist.to(device)
    gru_dist.eval()
    
    lstm_light = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_light.load_state_dict(torch.load("light_lstm.pt", map_location=device, weights_only=False))
    lstm_light.to(device)
    lstm_light.eval()
    
    lstm_opt = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_opt.load_state_dict(torch.load("optimized_window_lstm.pt", map_location=device, weights_only=False))
    lstm_opt.to(device)
    lstm_opt.eval()
    
    hybrid_xgb = XGBClassifier()
    hybrid_xgb.load_model("hybrid_xgb_model.json")
    
    # 3. Model Evaluations
    print("Evaluating models...")
    models_data = {}
    
    # LGB
    lgb_probs = lgb_base.predict_proba(X_val)[:, 1]
    models_data["Baseline LightGBM"] = lgb_probs
    
    # XGB
    xgb_probs = xgb_base.predict_proba(X_val)[:, 1]
    models_data["Advanced XGBoost"] = xgb_probs
    
    # Baseline LSTM
    lstm_base_probs = get_predictions(lstm_base, seqs_64, device)
    models_data["Baseline Champion LSTM"] = lstm_base_probs
    
    # Distilled GRU
    gru_probs = get_predictions(gru_dist, seqs_64, device)
    models_data["Distilled GRU Student"] = gru_probs
    
    # Lighter LSTM
    light_lstm_probs = get_predictions(lstm_light, seqs_64, device)
    models_data["Lighter LSTM Student"] = light_lstm_probs
    
    # Optimized Window LSTM
    opt_lstm_probs = get_predictions(lstm_opt, seqs_32, device)
    models_data["Optimized Window LSTM"] = opt_lstm_probs
    
    # Hybrid Stacking XGBoost
    hybrid_probs = hybrid_xgb.predict_proba(X_val_hybrid)[:, 1]
    models_data["Hybrid Stacking XGBoost"] = hybrid_probs
    
    # 4. Generate Confusion Matrices & Metrics Report
    print("\n" + "="*80)
    print("                 DETAILED CONFUSION MATRICES & PR-AUC REPORT                ")
    print("="*80)
    
    plt.figure(figsize=(10, 8))
    
    for name, probs in models_data.items():
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds)
        
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_val, probs)
        pr_auc = auc(recall, precision)
        
        # Plot PR curve
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
        print(f"\nModel: {name}")
        print(f"  - Accuracy: {acc*100:.4f}% | PR-AUC: {pr_auc:.4f}")
        print(f"  - False Positive Rate (FPR): {fpr*100:.4f}% ({fp}/{fp+tn})")
        print(f"  - False Negative Rate (FNR): {fnr*100:.4f}% ({fn}/{fn+tp})")
        print(f"  - Confusion Matrix:")
        print(f"        TN: {tn:<7} | FP: {fp}")
        print(f"        FN: {fn:<7} | TP: {tp}")
        print("-" * 50)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - All DGA Detection Models')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    plot_path = "precision_recall_curves.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nPrecision-Recall curves plot saved successfully to {plot_path}.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
