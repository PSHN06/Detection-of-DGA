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
from lightgbm import LGBMClassifier
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
    
    # 1. Load Master Checkpoints
    print("Loading all master checkpoints and caches...")
    
    vectorizer = joblib.load("ngram_vectorizer.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # Classical models
    lgb_model = joblib.load("baseline_lgb_model.pkl")
    xgb_model = XGBClassifier()
    xgb_model.load_model("advanced_xgb_model.json")
    
    # Baseline LSTM (Window=64, Hidden=64)
    lstm_base = LSTMClassifier(vocab_size=vocab_size, hidden_dim=64)
    lstm_base.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device, weights_only=False))
    lstm_base.eval()
    
    # Distilled GRU (Window=64, Hidden=64)
    gru_dist = GRUClassifier(vocab_size=vocab_size)
    gru_dist.load_state_dict(torch.load("distilled_gru.pt", map_location=device, weights_only=False))
    gru_dist.eval()
    
    # Lighter LSTM (Window=64, Hidden=32)
    lstm_light = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_light.load_state_dict(torch.load("light_lstm.pt", map_location=device, weights_only=False))
    lstm_light.eval()
    
    # Optimized Window LSTM (Window=32, Hidden=32)
    lstm_opt = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_opt.load_state_dict(torch.load("optimized_window_lstm.pt", map_location=device, weights_only=False))
    lstm_opt.eval()
    
    # Hybrid Stacking XGBoost
    hybrid_xgb = XGBClassifier()
    hybrid_xgb.load_model("hybrid_xgb_model.json")
    
    print("All checkpoints loaded successfully.")
    
    # Predefined column mapping
    cols = ['length', 'entropy', 'digit_density', 'vowel_consonant_ratio'] + [f"ngram_{i}" for i in range(500)]
    cols_augmented = cols + ['lstm_prob']
    
    def run_pipeline(domain_str):
        # 1. Unified Feature Extraction & Data Prep
        # Force domain string to lowercase
        domain_lower = domain_str.lower().strip()
        domain = get_sld(domain_lower)
        
        # Dense lexical and n-grams
        lex_feats = [len(domain), shannon_entropy(domain), digit_density(domain), vowel_to_consonant_ratio(domain)]
        ngram_arr = vectorizer.transform([domain]).toarray()[0]
        comb_feats = np.concatenate([lex_feats, ngram_arr]).reshape(1, -1)
        
        # Create DataFrames with column names matching scaler fit order
        scaled_df = pd.DataFrame(comb_feats, columns=cols)
        scaled_feats = scaler.transform(scaled_df)
        scaled_feats_df = pd.DataFrame(scaled_feats, columns=cols)
        
        # Sequence token tensors (mapping to vocabulary integers)
        def get_sequence_tensor(max_len):
            seq = [char_to_idx.get(c, 1) for c in domain]
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            return torch.tensor([seq], dtype=torch.long)
            
        seq_64 = get_sequence_tensor(64)
        seq_32 = get_sequence_tensor(32)
        
        # 2. Standalone Inference with timing
        # LightGBM
        start = time.perf_counter()
        lgb_prob = lgb_model.predict_proba(scaled_feats_df)[0, 1]
        lgb_lat = (time.perf_counter() - start) * 1e6
        
        # XGBoost
        start = time.perf_counter()
        xgb_prob = xgb_model.predict_proba(scaled_feats_df)[0, 1]
        xgb_lat = (time.perf_counter() - start) * 1e6
        
        # Baseline LSTM
        start = time.perf_counter()
        with torch.no_grad():
            lstm_base_prob = lstm_base(seq_64).item()
        lstm_base_lat = (time.perf_counter() - start) * 1e6
        
        # Distilled GRU
        start = time.perf_counter()
        with torch.no_grad():
            gru_dist_prob = gru_dist(seq_64).item()
        gru_dist_lat = (time.perf_counter() - start) * 1e6
        
        # Lighter LSTM
        start = time.perf_counter()
        with torch.no_grad():
            lstm_light_prob = lstm_light(seq_64).item()
        lstm_light_lat = (time.perf_counter() - start) * 1e6
        
        # Optimized Window LSTM
        start = time.perf_counter()
        with torch.no_grad():
            lstm_opt_prob = lstm_opt(seq_32).item()
        lstm_opt_lat = (time.perf_counter() - start) * 1e6
        
        # Hybrid Stacking XGBoost
        start = time.perf_counter()
        augmented_feats = np.hstack([scaled_feats, [[lstm_opt_prob]]])
        augmented_df = pd.DataFrame(augmented_feats, columns=cols_augmented)
        hybrid_prob = hybrid_xgb.predict_proba(augmented_df)[0, 1]
        hybrid_lat = (time.perf_counter() - start) * 1e6
        
        return (lgb_prob, lgb_lat, xgb_prob, xgb_lat,
                lstm_base_prob, lstm_base_lat,
                gru_dist_prob, gru_dist_lat,
                lstm_light_prob, lstm_light_lat,
                lstm_opt_prob, lstm_opt_lat,
                hybrid_prob, hybrid_lat)
                
    def evaluate_pipeline(domain_str):
        (lgb_prob, lgb_lat, xgb_prob, xgb_lat,
         lstm_base_prob, lstm_base_lat,
         gru_dist_prob, gru_dist_lat,
         lstm_light_prob, lstm_light_lat,
         lstm_opt_prob, lstm_opt_lat,
         hybrid_prob, hybrid_lat) = run_pipeline(domain_str)
        
        # Helper to format outputs
        def get_verdict_str(prob):
            return "MALICIOUS" if prob >= 0.5 else "SAFE"
            
        def get_conf_pct(prob):
            return prob * 100 if prob >= 0.5 else (1.0 - prob) * 100
            
        # Compile model data for Lowest Latency Target (comparing all standalone sequence/tree runs)
        model_runs = [
            ("LightGBM", lgb_lat, lgb_prob),
            ("XGBoost", xgb_lat, xgb_prob),
            ("Baseline Champion LSTM", lstm_base_lat, lstm_base_prob),
            ("Distilled GRU Student", gru_dist_lat, gru_dist_prob),
            ("Lighter LSTM Student", lstm_light_lat, lstm_light_prob),
            ("Optimized Window LSTM", lstm_opt_lat, lstm_opt_prob),
            ("Hybrid Stacking XGBoost", hybrid_lat, hybrid_prob)
        ]
        
        # Find lowest latency
        lowest_model_name, lowest_lat, lowest_prob = min(model_runs, key=lambda x: x[1])
        
        # Final production verdict consensus calculation
        votes = [
            int(lgb_prob >= 0.5),
            int(xgb_prob >= 0.5),
            int(lstm_base_prob >= 0.5),
            int(gru_dist_prob >= 0.5),
            int(lstm_light_prob >= 0.5),
            int(lstm_opt_prob >= 0.5),
            int(hybrid_prob >= 0.5)
        ]
        num_malicious = sum(votes)
        is_malicious_consensus = num_malicious >= 4
        
        final_verdict_label = "MALICIOUS DGA" if is_malicious_consensus else "SAFE"
        
        # Engineering consensus justification sentence
        if is_malicious_consensus:
            justification = f"Ensemble consensus reached: {num_malicious}/7 models detected character-level randomness or structural n-gram DGA signals."
        else:
            justification = f"Ensemble consensus reached: {7 - num_malicious}/7 models confirmed standard natural language patterns or known benign structures."
            
        # 4. Print Consolidation
        print("\n======================================================================")
        print("                  ALL-MODEL SHOWDOWN & PRODUCTION DGA GATEWAY DASHBOARD ")
        print("======================================================================")
        print(f"Domain Evaluated: {domain_str}")
        print("----------------------------------------------------------------------")
        print("[CLASSICAL TREE INFRASTRUCTURE - 504 FEATURES]")
        print(f"- Standalone LightGBM      -> {get_verdict_str(lgb_prob):<9} (Confidence: {get_conf_pct(lgb_prob):5.2f}%) | Latency: {lgb_lat:7.2f} \u00b5s")
        print(f"- Standalone XGBoost       -> {get_verdict_str(xgb_prob):<9} (Confidence: {get_conf_pct(xgb_prob):5.2f}%) | Latency: {xgb_lat:7.2f} \u00b5s")
        print("\n[DEEP SEQUENTIAL ARCHITECTURES]")
        print(f"- Baseline Champion LSTM   -> {get_verdict_str(lstm_base_prob):<9} (Confidence: {get_conf_pct(lstm_base_prob):5.2f}%) | Latency: {lstm_base_lat:7.2f} \u00b5s")
        print(f"- Distilled GRU Student    -> {get_verdict_str(gru_dist_prob):<9} (Confidence: {get_conf_pct(gru_dist_prob):5.2f}%) | Latency: {gru_dist_lat:7.2f} \u00b5s")
        print(f"- Lighter LSTM Student     -> {get_verdict_str(lstm_light_prob):<9} (Confidence: {get_conf_pct(lstm_light_prob):5.2f}%) | Latency: {lstm_light_lat:7.2f} \u00b5s")
        print(f"- Optimized Window LSTM    -> {get_verdict_str(lstm_opt_prob):<9} (Confidence: {get_conf_pct(lstm_opt_prob):5.2f}%) | Latency: {lstm_opt_lat:7.2f} \u00b5s")
        print("\n[HYBRID METRIC ENSEMBLE SYSTEM - 505 FEATURES]")
        print(f"- Hybrid Stacking XGBoost  -> {get_verdict_str(hybrid_prob):<9} (Confidence: {get_conf_pct(hybrid_prob):5.2f}%) | Latency: {hybrid_lat:7.2f} \u00b5s")
        print("----------------------------------------------------------------------")
        print(" TARGETED SYSTEM METRICS:")
        print("[A] HIGHLY ACCURATE DEPLOYMENT PREDICTION:")
        print(f"    Model: Optimized Window LSTM (Acc: 84.95%) -> Prediction: {get_verdict_str(lstm_opt_prob)} (Confidence: {get_conf_pct(lstm_opt_prob):.2f}%)")
        print("\n[B] LOWEST INFERENCE LATENCY PROFILE:")
        print(f"    Model: {lowest_model_name} -> Latency: {lowest_lat:.2f} \u00b5s | Prediction: {get_verdict_str(lowest_prob)}")
        print("\n FINAL COMBINED PRODUCTION VERDICT:")
        print(f"    The input string is officially flagged as {final_verdict_label}.")
        print(f"    {justification}")
        print("======================================================================\n")

    # Silent Warmup on Load
    for _ in range(10):
        _ = run_pipeline("google.com")

    if interactive:
        while True:
            try:
                domain_input = input("Enter domain (or 'exit' to quit): ").strip()
                if not domain_input:
                    continue
                if domain_input.lower() in ['exit', 'quit']:
                    print("Exiting master showdown dashboard.")
                    break
                evaluate_pipeline(domain_input)
            except KeyboardInterrupt:
                print("\nExiting master showdown dashboard.")
                break
    else:
        print("\n============================================================")
        print("         RUNNING MASTER SHOWDOWN DEMO (NON-INTERACTIVE)      ")
        print("============================================================")
        demo_domains = [
            "google.com",
            "wikipedia.org",
            "vzxqk",
            "earnestnessbiophysicalohax.com"
        ]
        for domain in demo_domains:
            evaluate_pipeline(domain)
            print("-" * 60)
            
        print("Demo completed successfully. Run step20_ultimate_dashboard.py to test live.")

if __name__ == "__main__":
    main()
