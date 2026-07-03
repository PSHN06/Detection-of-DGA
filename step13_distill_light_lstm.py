import os
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from step8_train_lstm import LSTMClassifier, DomainDataset, get_sld

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Vocab
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # 2. Load Teacher Model (hidden_dim=64)
    print("Loading Teacher LSTM model (hidden_dim=64)...")
    teacher_model = LSTMClassifier(vocab_size=vocab_size, hidden_dim=64)
    teacher_model.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device, weights_only=False))
    teacher_model.to(device)
    teacher_model.eval() # Teacher in eval mode
    
    # 3. Define Student Model (hidden_dim=32)
    print("Initializing Student Lighter LSTM model (hidden_dim=32)...")
    student_model = LSTMClassifier(vocab_size=vocab_size, hidden_dim=32)
    student_model.to(device)
    
    # 4. Load train.csv and preprocess
    print("Loading raw training dataset...")
    train_df = pd.read_csv("train.csv")
    train_df['sld'] = train_df['domain'].apply(get_sld)
    
    # Tokenize and pad
    max_len = 64
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
        
    print("Tokenizing training set...")
    X_train_seq = tokenize_and_pad(train_df['sld'])
    y_train = train_df['class'].values
    
    train_dataset = DomainDataset(X_train_seq, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # 5. Training loop
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Starting distillation training loop (3 epochs)...")
    start_time = time.time()
    student_model.train()
    
    for epoch in range(3):
        epoch_loss = 0.0
        for seqs, _ in train_loader:
            seqs = seqs.to(device)
            
            # Get teacher soft probabilities
            with torch.no_grad():
                teacher_probs = teacher_model(seqs).squeeze(1)
                
            # Get student probabilities
            optimizer.zero_grad()
            student_probs = student_model(seqs).squeeze(1)
            
            # Calculate distillation MSE loss
            loss = criterion(student_probs, teacher_probs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * seqs.size(0)
            
        print(f"  - Epoch {epoch+1}/3 - Distillation MSE Loss: {epoch_loss / len(train_dataset):.6f}")
        
    end_time = time.time()
    print(f"Distillation training completed in {end_time - start_time:.4f} seconds.")
    
    # 6. Save model checkpoint
    output_path = "light_lstm.pt"
    torch.save(student_model.state_dict(), output_path)
    print(f"Lighter LSTM student model weights saved to {output_path} successfully.")

if __name__ == "__main__":
    main()
