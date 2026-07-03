import os
import sys
import time
import math
import json
import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from collections import Counter
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# LSTM Model class definition (updated with padding mask)
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

# Lexical feature extraction functions (must match step5 exactly)
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
    return float(num_vowels) / (num_consonants + 1e-5)

def extract_single_features(domain, vectorizer, scaler):
    length = len(domain)
    entropy = shannon_entropy(domain)
    density = digit_density(domain)
    ratio = vowel_to_consonant_ratio(domain)
    lex_df = pd.DataFrame([[length, entropy, density, ratio]], columns=['length', 'entropy', 'digit_density', 'vowel_consonant_ratio'])
    
    # n-grams
    ngram_arr = vectorizer.transform([domain]).toarray()
    ngram_cols = [f"ngram_{i}" for i in range(ngram_arr.shape[1])]
    ngram_df = pd.DataFrame(ngram_arr, columns=ngram_cols)
    
    # combine
    combined_df = pd.concat([lex_df, ngram_df], axis=1)
    
    # scale
    scaled = scaler.transform(combined_df)
    scaled_df = pd.DataFrame(scaled, columns=combined_df.columns)
    return scaled_df

def main():
    parser = argparse.ArgumentParser(description="Real-Time DGA Detection Adapter")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive CLI loop")
    args = parser.parse_args()

    print("Loading models and vocabulary...")
    # Load RF model
    rf_model = joblib.load("baseline_rf_model.pkl")
    # Load XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.load_model("advanced_xgb_model.json")
    
    # Load LSTM vocab and model weights
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    lstm_model = LSTMClassifier(vocab_size=vocab_size)
    lstm_model.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=torch.device('cpu')))
    lstm_model.eval()

    # Load pre-fitted CountVectorizer and StandardScaler
    print("Loading CountVectorizer and StandardScaler checkpoints...")
    vectorizer = joblib.load("ngram_vectorizer.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    print("Scaler and Vectorizer loaded successfully.")

    def run_inference(domain):
        print(f"\nAnalyzing domain: {domain}")
        print("-" * 50)
        
        # 1. Tabular features (RF & XGBoost)
        scaled_feats_df = extract_single_features(domain, vectorizer, scaler)
        
        # RF predict
        rf_prob = rf_model.predict_proba(scaled_feats_df)[0, 1]
        
        # XGBoost predict
        xgb_prob = xgb_model.predict_proba(scaled_feats_df)[0, 1]
        
        # 2. LSTM text features
        max_len = 64
        seq = [char_to_idx.get(c, 1) for c in domain]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        
        seq_tensor = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            lstm_prob = lstm_model(seq_tensor).item()
            
        # Print model evaluations
        for model_name, prob in [("Random Forest", rf_prob), ("XGBoost", xgb_prob), ("LSTM (Sequence)", lstm_prob)]:
            is_dga = prob >= 0.5
            status = "[ALERT] DGA Detected!" if is_dga else "[SAFE] Benign Domain"
            print(f"{model_name:<18} -> {status:<25} (Confidence: {prob*100:6.2f}%)")
        print("-" * 50)

    # Mode selection
    if args.interactive:
        print("\n" + "="*60)
        print("           REAL-TIME DGA DETECTION ADAPTER ACTIVE          ")
        print("="*60)
        print("Enter a domain name to analyze (or type 'exit' to quit):")
        
        while True:
            try:
                domain = input("Domain> ").strip()
                if not domain:
                    continue
                if domain.lower() in ['exit', 'quit']:
                    print("Exiting real-time adapter. Goodbye!")
                    break
                run_inference(domain)
            except KeyboardInterrupt:
                print("\nExiting real-time adapter. Goodbye!")
                break
            except Exception as e:
                print(f"Error processing domain: {e}")
    else:
        print("\n" + "="*60)
        print("             RUNNING DEMO MODE (NON-INTERACTIVE)           ")
        print("="*60)
        demo_domains = [
            "google.com",
            "wikipedia.org",
            "vzxqk",
            "earnestnessbiophysicalohax.com"
        ]
        for domain in demo_domains:
            run_inference(domain)
        print("\nDemo Mode completed successfully. Use --interactive flag to run manually.")

if __name__ == "__main__":
    main()
