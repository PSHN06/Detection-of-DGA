import os
import sys
import time
import json
import math
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from xgboost import XGBClassifier
from step8_train_lstm import LSTMClassifier, get_sld
from step14_distill_gru import GRUClassifier

def shannon_entropy(s):
    if not s or not isinstance(s, str):
        return 0.0
    counts = Counter(s)
    entropy = 0.0
    total = len(s)
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def digit_density(s):
    if not s or not isinstance(s, str):
        return 0.0
    digit_count = sum(1 for c in s if c.isdigit())
    return digit_count / len(s)

def vowel_to_consonant_ratio(s):
    if not s or not isinstance(s, str):
        return 0.0
    s_lower = s.lower()
    vowels = set('aeiou')
    consonants = set('bcdfghjklmnpqrstvwxyz')
    num_vowels = sum(1 for c in s_lower if c in vowels)
    num_consonants = sum(1 for c in s_lower if c in consonants)
    return float(num_vowels) / (num_consonants + 1)

def main():
    device = torch.device('cpu')
    
    # Check flags: default to interactive unless --demo is passed or stdin is not a tty
    interactive = "--demo" not in sys.argv and sys.stdin.isatty()
    
    # 1. Load Checkpoints & Preprocessing pipelines
    print("Loading models and preprocessing pipelines...")
    
    vectorizer = joblib.load("ngram_vectorizer.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    
    lgb_model = joblib.load("baseline_lgb_model.pkl")
    
    xgb_model = XGBClassifier()
    xgb_model.load_model("advanced_xgb_model.json")
    
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # Baseline LSTM (FP32, hidden=64)
    lstm_base = LSTMClassifier(vocab_size=vocab_size, hidden_dim=64)
    lstm_base.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device, weights_only=False))
    lstm_base.eval()
    
    # Distilled GRU (FP32, hidden=64)
    gru_dist = GRUClassifier(vocab_size=vocab_size)
    gru_dist.load_state_dict(torch.load("distilled_gru.pt", map_location=device, weights_only=False))
    gru_dist.eval()
    
    # Lighter LSTM (FP32, hidden=32)
    lstm_light = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_light.load_state_dict(torch.load("light_lstm.pt", map_location=device, weights_only=False))
    lstm_light.eval()
    
    # Optimized Window LSTM (FP32, hidden=32)
    lstm_opt = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_opt.load_state_dict(torch.load("optimized_window_lstm.pt", map_location=device, weights_only=False))
    lstm_opt.eval()
    
    print("All checkpoints loaded successfully.")
    
    def evaluate_all(domain_str):
        sld = get_sld(domain_str)
        
        # Preprocessing for Classical ML
        lex_feats = [len(sld), shannon_entropy(sld), digit_density(sld), vowel_to_consonant_ratio(sld)]
        ngram_arr = vectorizer.transform([sld]).toarray()[0]
        comb_feats = np.concatenate([lex_feats, ngram_arr]).reshape(1, -1)
        cols = ['length', 'entropy', 'digit_density', 'vowel_consonant_ratio'] + [f"ngram_{i}" for i in range(len(ngram_arr))]
        scaled_df = pd.DataFrame(comb_feats, columns=cols)
        scaled_feats = scaler.transform(scaled_df)
        scaled_feats_df = pd.DataFrame(scaled_feats, columns=cols)
        
        # 1. LightGBM
        start = time.perf_counter()
        lgb_prob = lgb_model.predict_proba(scaled_feats_df)[0, 1]
        lgb_lat = (time.perf_counter() - start) * 1e6
        
        # 2. XGBoost
        start = time.perf_counter()
        xgb_prob = xgb_model.predict_proba(scaled_feats_df)[0, 1]
        xgb_lat = (time.perf_counter() - start) * 1e6
        
        # Preprocessing for Sequence Models
        def get_sequence(max_len):
            seq = [char_to_idx.get(c, 1) for c in sld]
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            return torch.tensor([seq], dtype=torch.long)
            
        seq_64 = get_sequence(64)
        seq_32 = get_sequence(32)
        
        # 3. Baseline LSTM
        start = time.perf_counter()
        with torch.no_grad():
            base_prob = lstm_base(seq_64).item()
        base_lat = (time.perf_counter() - start) * 1e6
        
        # 4. Distilled GRU
        start = time.perf_counter()
        with torch.no_grad():
            gru_prob = gru_dist(seq_64).item()
        gru_lat = (time.perf_counter() - start) * 1e6
        
        # 5. Lighter LSTM
        start = time.perf_counter()
        with torch.no_grad():
            light_prob = lstm_light(seq_64).item()
        light_lat = (time.perf_counter() - start) * 1e6
        
        # 6. Optimized Window LSTM
        start = time.perf_counter()
        with torch.no_grad():
            opt_prob = lstm_opt(seq_32).item()
        opt_lat = (time.perf_counter() - start) * 1e6
        
        # Format prediction helper
        def get_pred_label(prob):
            return "[MALICIOUS DGA]" if prob >= 0.5 else "[SAFE/BENIGN]"
            
        print("\n======================================================================")
        print("LIVE DGA DETECTION DASHBOARD (Interactive Mode)")
        print("======================================================================")
        print(f"Domain Input: {domain_str}")
        print("----------------------------------------------------------------------")
        print("[CLASSICAL ML CORE]")
        print(f"- LightGBM       -> Prediction: {get_pred_label(lgb_prob):<16} \n(Confidence: {lgb_prob*100:5.2f}%) | Latency: {lgb_lat:6.2f} \u00b5s")
        print(f"- XGBoost        -> Prediction: {get_pred_label(xgb_prob):<16} \n(Confidence: {xgb_prob*100:5.2f}%) | Latency: {xgb_lat:6.2f} \u00b5s")
        print("\n[BASELINE DEEP LEARNING CELL]")
        print(f"- Baseline LSTM  -> Prediction: {get_pred_label(base_prob):<16} \n(Confidence: {base_prob*100:5.2f}%) | Latency: {base_lat:6.2f} \u00b5s")
        print("\n[LATENCY OPTIMIZATION VARIANTS]")
        print(f"- Distilled GRU  -> Prediction: {get_pred_label(gru_prob):<16} \n(Confidence: {gru_prob*100:5.2f}%) | Latency: {gru_lat:6.2f} \u00b5s")
        print(f"- Lighter LSTM   -> Prediction: {get_pred_label(light_prob):<16} \n(Confidence: {light_prob*100:5.2f}%) | Latency: {light_lat:6.2f} \u00b5s")
        print("Tighter-Window LSTM (32-char Window, 32-Hidden Dim)")
        print(f"  --> Prediction: {'[ALERT] MALICIOUS DGA DETECTED!' if opt_prob >= 0.5 else '[SAFE] SAFE / BENIGN'}")
        print(f"  --> Confidence: {opt_prob*100:.2f}%")
        print(f"  --> Inference Latency: {opt_lat:.2f} \u00b5s")
        print("======================================================================\n")

    if interactive:
        while True:
            try:
                domain_input = input("Enter domain (or 'quit' to exit): ").strip()
                if not domain_input:
                    continue
                if domain_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                evaluate_all(domain_input)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        print("\n--- Running Master Dashboard Demo ---")
        evaluate_all("vzxqk")
        evaluate_all("google.com")

if __name__ == "__main__":
    main()
