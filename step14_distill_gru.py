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

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, dropout_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lengths = torch.clamp((x != 0).sum(dim=1), min=1).cpu()
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_gru_out, _ = self.gru(packed_embedded)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_gru_out, batch_first=True, total_length=x.size(1)
        )
        mask = (x == 0).unsqueeze(-1)
        masked_gru_out = gru_out.masked_fill(mask, -1e9)
        last_out, _ = torch.max(masked_gru_out, dim=1)
        out = self.dropout(last_out)
        out = self.fc(out)
        return self.sigmoid(out)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Vocab
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    
    # 2. Load Teacher Model
    print("Loading Teacher LSTM model...")
    teacher_model = LSTMClassifier(vocab_size=vocab_size)
    teacher_model.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=device))
    teacher_model.to(device)
    teacher_model.eval() # Teacher must be in eval mode
    
    # 3. Define Student Model
    print("Initializing Student GRU model...")
    student_model = GRUClassifier(vocab_size=vocab_size)
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
    # Distillation loss: MSE Loss between student probability and teacher probability
    criterion = nn.MSELoss()
    
    print("Starting distillation training loop (3 epochs)...")
    start_time = time.time()
    student_model.train()
    
    for epoch in range(3):
        epoch_loss = 0.0
        for seqs, _ in train_loader:
            seqs = seqs.to(device)
            
            # Get teacher soft probabilities (no_grad)
            with torch.no_grad():
                teacher_probs = teacher_model(seqs).squeeze(1)
                
            # Get student probabilities
            optimizer.zero_grad()
            student_probs = student_model(seqs).squeeze(1)
            
            # Calculate MSE loss comparing student to teacher
            loss = criterion(student_probs, teacher_probs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * seqs.size(0)
            
        print(f"  - Epoch {epoch+1}/3 - Distillation MSE Loss: {epoch_loss / len(train_dataset):.6f}")
        
    end_time = time.time()
    print(f"Distillation training completed in {end_time - start_time:.4f} seconds.")
    
    # 6. Save model checkpoint
    output_path = "distilled_gru.pt"
    torch.save(student_model.state_dict(), output_path)
    print(f"Distilled GRU model weights saved to {output_path} successfully.")

if __name__ == "__main__":
    main()
