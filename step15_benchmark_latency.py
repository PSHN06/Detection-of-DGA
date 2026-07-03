import os
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from step8_train_lstm import LSTMClassifier, get_sld
from step14_distill_gru import GRUClassifier

def compute_fnr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0

def main():
    device = torch.device('cpu')
    print("Benchmarking on CPU to ensure fair baseline comparison.")
    
    # 1. Load Vocab
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # 2. Load Checkpoints
    print("Loading champion LSTM model (hidden_dim=64)...")
    lstm_model = LSTMClassifier(vocab_size=vocab_size, hidden_dim=64)
    lstm_model.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device, weights_only=False))
    lstm_model.eval()
    
    print("Loading Lighter LSTM model (hidden_dim=32)...")
    light_lstm = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    light_lstm.load_state_dict(torch.load("light_lstm.pt", map_location=device, weights_only=False))
    light_lstm.eval()
    
    print("Loading distilled GRU model...")
    gru_model = GRUClassifier(vocab_size=vocab_size)
    gru_model.load_state_dict(torch.load("distilled_gru.pt", map_location=device, weights_only=False))
    gru_model.eval()
    
    # 3. Load Test Data Subset (Dataset B)
    print("Loading subset from dataset_b_external.csv...")
    df_b = pd.read_csv("dataset_b_external.csv")
    subset = df_b.sample(2000, random_state=42)
    y_true = subset['class'].values
    
    # Tokenize and pad
    max_len = 64
    def tokenize_and_pad(domains):
        sequences = []
        for domain in domains:
            domain_str = get_sld(domain)
            seq = [char_to_idx.get(c, 1) for c in domain_str]
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            sequences.append(seq)
        return np.array(sequences)
        
    X_seqs = tokenize_and_pad(subset['domain'])
    
    # 4. Benchmarking function
    def benchmark_model(model, name, file_path):
        print(f"Benchmarking {name}...")
        latencies = []
        preds = []
        
        # Warmup
        warmup_tensor = torch.zeros((1, 64), dtype=torch.long)
        for _ in range(100):
            with torch.no_grad():
                _ = model(warmup_tensor)
                
        # Main benchmark
        for seq in X_seqs:
            seq_tensor = torch.tensor([seq], dtype=torch.long)
            
            start = time.perf_counter()
            with torch.no_grad():
                out = model(seq_tensor)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1e6) # microseconds
            preds.append(int(out.item() >= 0.5))
            
        preds = np.array(preds)
        acc = accuracy_score(y_true, preds)
        fnr = compute_fnr(y_true, preds)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        model_size = os.path.getsize(file_path) / (1024 * 1024)
        
        return {
            'Name': name,
            'Size (MB)': model_size,
            'Avg Latency (µs)': avg_latency,
            '95th Percentile Latency (µs)': p95_latency,
            'Accuracy': acc,
            'FNR': fnr
        }
        
    # Run benchmarks
    results = []
    results.append(benchmark_model(lstm_model, "Champion LSTM (FP32)", "advanced_lstm_model.pt"))
    results.append(benchmark_model(light_lstm, "Lighter LSTM (FP32)", "light_lstm.pt"))
    results.append(benchmark_model(gru_model, "Distilled GRU (FP32)", "distilled_gru.pt"))
    
    df_results = pd.DataFrame(results)
    
    # 5. Print Comparison Table
    print("\n" + "="*95)
    print("                       LATENCY & PERFORMANCE BENCHMARK SUMMARY                       ")
    print("="*95)
    print(f"| {'Model Name':<23} | {'Size (MB)':<10} | {'Avg Latency (µs)':<17} | {'p95 Latency (µs)':<17} | {'Accuracy':<10} | {'FNR':<8} |")
    print("-"*95)
    for res in results:
        print(f"| {res['Name']:<23} | {res['Size (MB)']:<10.6f} | {res['Avg Latency (µs)']:<17.2f} | {res['95th Percentile Latency (µs)']:<17.2f} | {res['Accuracy']:<10.6f} | {res['FNR']:<8.6f} |")
    print("="*95 + "\n")
    
    # 6. Generate Visual Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    scatter = sns.scatterplot(
        data=df_results,
        x='Avg Latency (µs)',
        y='Accuracy',
        hue='Name',
        s=150,
        palette='Set1',
        edgecolor='0.2'
    )
    
    # Annotate points with sizes and FNR
    for idx, row in df_results.iterrows():
        plt.text(
            row['Avg Latency (µs)'] + 0.5,
            row['Accuracy'] + 0.001,
            f"Size: {row['Size (MB)']:.3f}MB\nFNR: {row['FNR']*100:.2f}%",
            fontsize=9,
            fontweight='bold',
            verticalalignment='bottom'
        )
        
    plt.title("Sequence Model Trade-off: Inference Latency vs. Generalization Accuracy", fontsize=13, fontweight='bold')
    plt.xlabel("Average Inference Latency (microseconds / sample)")
    plt.ylabel("Classification Accuracy on Dataset B (External)")
    plt.grid(True, which="both", ls="--", c="0.7")
    plt.tight_layout()
    plt.savefig('latency_vs_performance.png', dpi=300)
    plt.close()
    print("Scatter visualization saved as latency_vs_performance.png.")

if __name__ == "__main__":
    main()
