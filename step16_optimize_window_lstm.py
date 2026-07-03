import os
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from step8_train_lstm import LSTMClassifier, DomainDataset, get_sld

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def compute_fnr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Vocab
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # 2. Analyze Optimal Length
    print("Loading train.csv to analyze optimal length...")
    train_df = pd.read_csv("train.csv")
    raw_lengths = train_df['domain'].astype(str).apply(len)
    optimal_max_len = int(np.percentile(raw_lengths, 99))
    print(f"\n==================================================")
    print(f"99th Percentile of Domain Length: {np.percentile(raw_lengths, 99)}")
    print(f"Optimal Max Window Length Configured: {optimal_max_len}")
    print(f"==================================================\n")
    
    # 3. Load Teacher (hidden_dim=64)
    print("Loading Teacher LSTM model...")
    teacher_model = LSTMClassifier(vocab_size=vocab_size, hidden_dim=64)
    teacher_model.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device, weights_only=False))
    teacher_model.to(device)
    teacher_model.eval()
    
    # 4. Initialize Student (hidden_dim=32, window=optimal_max_len)
    print(f"Initializing Student Lighter LSTM (hidden_dim=32, window={optimal_max_len})...")
    student_model = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    student_model.to(device)
    
    # Preprocess train set SLDs
    train_df['sld'] = train_df['domain'].apply(get_sld)
    
    # Tokenizer helper
    def tokenize_and_pad(domains, max_len):
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
        
    print("Tokenizing training set with new optimal max length...")
    X_train_seq = tokenize_and_pad(train_df['sld'], max_len=optimal_max_len)
    # Teacher needs max_len=64 tokenization
    X_train_seq_teacher = tokenize_and_pad(train_df['sld'], max_len=64)
    
    # Custom Dataset to feed both inputs
    class DualDomainDataset(torch.utils.data.Dataset):
        def __init__(self, seqs_student, seqs_teacher):
            self.seqs_student = torch.tensor(seqs_student, dtype=torch.long)
            self.seqs_teacher = torch.tensor(seqs_teacher, dtype=torch.long)
        def __len__(self):
            return len(self.seqs_student)
        def __getitem__(self, idx):
            return self.seqs_student[idx], self.seqs_teacher[idx]
            
    train_dataset = DualDomainDataset(X_train_seq, X_train_seq_teacher)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # 5. Distillation loop
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Starting distillation training loop (3 epochs)...")
    start_time = time.time()
    student_model.train()
    
    for epoch in range(3):
        epoch_loss = 0.0
        for seqs_student, seqs_teacher in train_loader:
            seqs_student = seqs_student.to(device)
            seqs_teacher = seqs_teacher.to(device)
            
            # Get teacher soft probabilities
            with torch.no_grad():
                teacher_probs = teacher_model(seqs_teacher).squeeze(1)
                
            # Get student probabilities
            optimizer.zero_grad()
            student_probs = student_model(seqs_student).squeeze(1)
            
            loss = criterion(student_probs, teacher_probs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * seqs_student.size(0)
            
        print(f"  - Epoch {epoch+1}/3 - Distillation MSE Loss: {epoch_loss / len(train_dataset):.6f}")
        
    end_time = time.time()
    print(f"Distillation training completed in {end_time - start_time:.4f} seconds.")
    
    # 6. Save Checkpoint
    output_path = "optimized_window_lstm.pt"
    torch.save(student_model.state_dict(), output_path)
    print(f"Optimized window student LSTM weights saved to {output_path} successfully.")
    
    # 7. Internal Benchmarking
    print("\nLoading test partition from dataset_b_external.csv for benchmarking...")
    df_b = pd.read_csv("dataset_b_external.csv")
    subset = df_b.sample(2000, random_state=42)
    y_true = subset['class'].values
    
    cpu_device = torch.device('cpu')
    
    # Load light_lstm for baseline reference
    print("Loading Lighter LSTM baseline model...")
    light_lstm = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    light_lstm.load_state_dict(torch.load("light_lstm.pt", map_location=cpu_device, weights_only=False))
    light_lstm.eval()
    
    # Set student to cpu and eval mode
    student_model.to(cpu_device)
    student_model.eval()
    
    # Tokenize test sets
    X_test_64 = tokenize_and_pad(subset['domain'].apply(get_sld), max_len=64)
    X_test_opt = tokenize_and_pad(subset['domain'].apply(get_sld), max_len=optimal_max_len)
    
    def benchmark(model, X_seqs, name):
        print(f"Benchmarking {name}...")
        latencies = []
        preds = []
        
        # Warmup
        warmup_tensor = torch.zeros((1, X_seqs.shape[1]), dtype=torch.long)
        for _ in range(100):
            with torch.no_grad():
                _ = model(warmup_tensor)
                
        # Run test
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
        
        return avg_latency, acc, fnr
        
    lat_light, acc_light, fnr_light = benchmark(light_lstm, X_test_64, "Lighter LSTM (window=64)")
    lat_opt, acc_opt, fnr_opt = benchmark(student_model, X_test_opt, "Optimized Window LSTM (window=33)")
    
    print("\n" + "="*85)
    print("                     INTERNAL LATENCY & WINDOW COMPARISON                     ")
    print("="*85)
    print(f"| {'Model Name':<28} | {'Window Size':<12} | {'Avg Latency (µs)':<17} | {'Accuracy':<10} | {'FNR':<8} |")
    print("-"*85)
    print(f"| {'Lighter LSTM baseline':<28} | {'64':<12} | {lat_light:<17.2f} | {acc_light:<10.6f} | {fnr_light:<8.6f} |")
    print(f"| {'Optimized Window LSTM':<28} | {str(optimal_max_len):<12} | {lat_opt:<17.2f} | {acc_opt:<10.6f} | {fnr_opt:<8.6f} |")
    print(f"| {'Delta Speedup (%)':<28} | {'N/A':<12} | {(1.0 - lat_opt/lat_light)*100:<17.2f}% | {'N/A':<10} | {'N/A':<8} |")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()

