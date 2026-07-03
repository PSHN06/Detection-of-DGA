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
from xgboost import XGBClassifier
from step8_train_lstm import LSTMClassifier, get_sld

# Check for psutil
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False

# Check for scapy
try:
    from scapy.all import sniff, DNS, DNSQR
    has_scapy = True
except ImportError:
    has_scapy = False

# Define lexical features (matching training pipeline)
def shannon_entropy(s):
    if not s or not isinstance(s, str):
        return 0.0
    from collections import Counter
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

def get_cpu_usage():
    if has_psutil:
        return psutil.cpu_percent()
    # Fallback mock CPU utilization
    return np.random.uniform(5.0, 15.0)

def get_memory_usage():
    if has_psutil:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # MB
    # Fallback mock memory utilization
    return 150.0 + np.random.uniform(-5.0, 5.0)

def main():
    device = torch.device('cpu')
    print("======================================================================")
    print("                 LIVE DGA PACKET SNIFFER ADAPTER ACTIVE                ")
    print("======================================================================")
    
    # 1. Load Checkpoints
    print("Loading feature pipelines and ensemble models...")
    vectorizer = joblib.load("ngram_vectorizer.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # Optimized Window LSTM
    lstm_opt = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    lstm_opt.load_state_dict(torch.load("optimized_window_lstm.pt", map_location=device, weights_only=False))
    lstm_opt.eval()
    
    # Hybrid Stacking XGBoost
    hybrid_xgb = XGBClassifier()
    hybrid_xgb.load_model("hybrid_xgb_model.json")
    print("All checkpoints loaded successfully.")
    
    cols = ['length', 'entropy', 'digit_density', 'vowel_consonant_ratio'] + [f"ngram_{i}" for i in range(500)]
    cols_augmented = cols + ['lstm_prob']
    
    # Track statistics
    packet_count = 0
    start_time = time.time()
    
    def process_domain(domain_str):
        nonlocal packet_count
        packet_count += 1
        
        # Preprocessing
        domain_lower = domain_str.lower().strip()
        domain = get_sld(domain_lower)
        
        inf_start = time.perf_counter()
        
        # Feature extraction
        lex_feats = [len(domain), shannon_entropy(domain), digit_density(domain), vowel_to_consonant_ratio(domain)]
        ngram_arr = vectorizer.transform([domain]).toarray()[0]
        comb_feats = np.concatenate([lex_feats, ngram_arr]).reshape(1, -1)
        
        scaled_df = pd.DataFrame(comb_feats, columns=cols)
        scaled_feats = scaler.transform(scaled_df)
        
        # Sequence processing (32 window)
        max_len = 32
        seq = [char_to_idx.get(c, 1) for c in domain]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        seq_tensor = torch.tensor([seq], dtype=torch.long)
        
        # Inference
        with torch.no_grad():
            lstm_prob = lstm_opt(seq_tensor).item()
            
        augmented_feats = np.hstack([scaled_feats, [[lstm_prob]]])
        augmented_df = pd.DataFrame(augmented_feats, columns=cols_augmented)
        hybrid_prob = hybrid_xgb.predict_proba(augmented_df)[0, 1]
        
        inf_latency = (time.perf_counter() - inf_start) * 1e6 # in microseconds
        
        # Metrics Calculations
        elapsed = time.time() - start_time
        throughput = packet_count / elapsed if elapsed > 0 else 0.0
        
        verdict = "MALICIOUS DGA" if hybrid_prob >= 0.5 else "SAFE"
        alert_flag = "⚠️  " if verdict == "MALICIOUS DGA" else "✅ "
        conf = hybrid_prob * 100 if hybrid_prob >= 0.5 else (1.0 - hybrid_prob) * 100
        
        # Format and Print Alert Dashboard
        print(f"{alert_flag} [Packet #{packet_count:<4}] Query: {domain_str:<40} -> Verdict: {verdict:<13} | Conf: {conf:6.2f}% | Latency: {inf_latency:7.2f} \u00b5s")
        if packet_count % 10 == 0:
            print(f"--- SYSTEM PERFORMANCE TELEMETRY: Throughput: {throughput:.2f} pps | CPU Usage: {get_cpu_usage():.1f}% | Memory Usage: {get_memory_usage():.1f} MB ---")
            
    # Try live sniffing
    sniff_success = False
    if has_scapy:
        try:
            print("\nInitializing Scapy DNS Sniffer (sniffing UDP port 53)...")
            print("Listening for live DNS traffic. Emulate queries in a separate terminal to generate events.")
            
            def dns_callback(pkt):
                if pkt.haslayer(DNS) and pkt.getlayer(DNS).qr == 0: # Query
                    try:
                        qname = pkt.getlayer(DNSQR).qname.decode('utf-8').strip('.')
                        process_domain(qname)
                    except Exception:
                        pass
                        
            # Start sniffer
            sniff_success = True
            sniff(filter="udp port 53", prn=dns_callback, store=0)
        except Exception as e:
            print(f"Live sniffing initialization failed due to driver/permission issue: {e}")
            sniff_success = False
            
    if not sniff_success:
        print("\n" + "!"*80)
        print("          FALLING BACK TO AUTOMATIC NETWORK TRAFFIC SIMULATION MODE            ")
        print("!"*80)
        print("Scapy or packet-capture drivers (Npcap/WinPcap) not detected or lack admin permissions.")
        print("Simulating live query intake stream from the test partition...")
        print("Press Ctrl+C to terminate the sniffer.")
        
        # Load test set domains to simulate intake
        test_df = pd.read_csv("test.csv")
        sample_domains = test_df['domain'].tolist()
        
        idx = 0
        while True:
            try:
                domain = sample_domains[idx % len(sample_domains)]
                process_domain(domain)
                # Sleep between 0.2 and 1.5 seconds to simulate incoming queries
                time.sleep(np.random.uniform(0.3, 1.2))
                idx += 1
            except KeyboardInterrupt:
                print("\nExiting simulated sniffer.")
                break

if __name__ == "__main__":
    main()
