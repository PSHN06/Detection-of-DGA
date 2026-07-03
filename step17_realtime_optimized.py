import os
import sys
import time
import json
import torch
import torch.nn as nn
from step8_train_lstm import LSTMClassifier, get_sld

def main():
    device = torch.device('cpu')
    
    # 1. Load Vocab
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # 2. Load optimized_window_lstm model
    print("Loading optimized window LSTM model (hidden_dim=32, max_len=33)...")
    model = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    model.load_state_dict(torch.load("optimized_window_lstm.pt", map_location=device, weights_only=False))
    model.eval()
    
    max_len = 33
    
    def predict_domain(domain_str):
        # 3. Live Tokenization & Padding (after TLD stripping)
        sld = get_sld(domain_str)
        seq = [char_to_idx.get(c, 1) for c in sld]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
            
        seq_tensor = torch.tensor([seq], dtype=torch.long)
        
        # 4. Instant Inference with Timing
        start = time.perf_counter()
        with torch.no_grad():
            out = model(seq_tensor)
        end = time.perf_counter()
        
        elapsed_us = (end - start) * 1e6
        prob = out.item()
        
        return prob, elapsed_us
        
    # Check flags
    interactive = "--interactive" in sys.argv
    
    if interactive:
        print("\n" + "="*60)
        print("          OPTIMIZED REAL-TIME DGA DETECTION GATEWAY          ")
        print("="*60)
        print("Type 'exit' or 'quit' to close the interface.\n")
        
        while True:
            try:
                domain_input = input("Enter domain: ").strip()
                if not domain_input:
                    continue
                if domain_input.lower() in ['exit', 'quit']:
                    print("Exiting real-time gateway.")
                    break
                    
                prob, elapsed_us = predict_domain(domain_input)
                
                print(f"  - Inference Latency: {elapsed_us:.2f} microseconds")
                if prob >= 0.5:
                    print(f"  - Prediction: \033[91m[ALERT] MALICIOUS DGA DETECTED (Confidence: {prob*100:.2f}%)\033[0m")
                else:
                    print(f"  - Prediction: \033[92m[SAFE] SAFE / BENIGN (Confidence: {(1.0 - prob)*100:.2f}%)\033[0m")
                print("-" * 60)
            except KeyboardInterrupt:
                print("\nExiting real-time gateway.")
                break
    else:
        print("\n============================================================")
        print("             RUNNING DEMO MODE (NON-INTERACTIVE)           ")
        print("============================================================")
        demo_domains = [
            "google.com",
            "wikipedia.org",
            "vzxqk",
            "earnestnessbiophysicalohax.com"
        ]
        for domain in demo_domains:
            prob, elapsed_us = predict_domain(domain)
            print(f"Analyzing domain: {domain}")
            print(f"  - Inference Latency: {elapsed_us:.2f} microseconds")
            if prob >= 0.5:
                print(f"  - Prediction: [ALERT] MALICIOUS DGA DETECTED (Confidence: {prob*100:.2f}%)")
            else:
                print(f"  - Prediction: [SAFE] SAFE / BENIGN (Confidence: {(1.0 - prob)*100:.2f}%)")
            print("-" * 60)
            
        print("Demo Mode completed successfully. Use --interactive flag to run manually.")

if __name__ == "__main__":
    main()
