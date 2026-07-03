import os
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DomainDataset(Dataset):
    def __init__(self, tokenized_sequences, labels):
        self.sequences = torch.tensor(tokenized_sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

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

def get_sld(d):
    d_str = str(d).lower().strip()
    if d_str.startswith('www.'):
        d_str = d_str[4:]
    return d_str.split('.')[0]

def main():
    # 1. Load raw text datasets
    print("Loading raw text datasets...")
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    
    # Preprocess to get SLDs
    train_df['sld'] = train_df['domain'].apply(get_sld)
    val_df['sld'] = val_df['domain'].apply(get_sld)
    
    print(f"Dataset sizes:")
    print(f"  - Train: {train_df.shape[0]} rows")
    print(f"  - Val:   {val_df.shape[0]} rows")
    
    # 2. Build character vocabulary mapping (tokenizer)
    print("Building character vocabulary...")
    all_text = "".join(train_df['sld'].astype(str).tolist())
    chars = sorted(list(set(all_text)))
    
    char_to_idx = {char: idx + 2 for idx, char in enumerate(chars)}
    char_to_idx['<pad>'] = 0
    char_to_idx['<unk>'] = 1
    
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size} unique tokens (including <pad> and <unk>)")
    
    with open("lstm_vocab.json", "w", encoding="utf-8") as f:
        json.dump(char_to_idx, f)
    print("Character vocabulary saved to lstm_vocab.json.")
    
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
        
    print("Tokenizing datasets...")
    X_train_seq = tokenize_and_pad(train_df['sld'])
    X_val_seq = tokenize_and_pad(val_df['sld'])
    
    y_train = train_df['class'].values
    y_val = val_df['class'].values
    
    # 3. Create PyTorch DataLoaders
    train_dataset = DomainDataset(X_train_seq, y_train)
    val_dataset = DomainDataset(X_val_seq, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # 4. Initialize model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, dropout_prob=0.3)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Train the model
    print("Training character-level LSTM model...")
    num_epochs = 3
    start_train = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * seqs.size(0)
            
        avg_loss = epoch_loss / len(train_dataset)
        print(f"  - Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
    end_train = time.time()
    train_time = end_train - start_train
    print(f"Training completed in {train_time:.4f} seconds.")
    
    # 6. Inference on validation set
    print("Running inference on the validation set...")
    model.eval()
    val_preds = []
    val_probs = []
    
    start_inf = time.time()
    with torch.no_grad():
        for seqs, _ in val_loader:
            seqs = seqs.to(device)
            outputs = model(seqs).squeeze(1)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            val_probs.extend(probs)
            val_preds.extend(preds)
            
    end_inf = time.time()
    total_inf_time = end_inf - start_inf
    latency_per_sample = total_inf_time / len(y_val)
    
    val_preds = np.array(val_preds)
    val_probs = np.array(val_probs)
    
    # 7. Calculate validation metrics
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds)
    rec = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    
    cm = confusion_matrix(y_val, val_preds)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Print metrics
    print("\n" + "="*50)
    print("           ADVANCED LSTM MODEL EVALUATION REPORT           ")
    print("="*50)
    print(f"Training Time:      {train_time:.4f} seconds")
    print(f"Inference Latency:  {latency_per_sample * 1e6:.4f} microseconds per sample")
    print("-"*50)
    print("PERFORMANCE METRICS:")
    print(f"  - Accuracy:       {acc:.6f}")
    print(f"  - Precision:      {prec:.6f}")
    print(f"  - Recall:         {rec:.6f}")
    print(f"  - F1 Score:       {f1:.6f}")
    print(f"  - ROC-AUC:        {roc_auc:.6f}")
    print(f"  - PR-AUC:         {pr_auc:.6f}")
    print(f"  - FPR:            {fpr:.6f} ({fp}/{fp+tn})")
    print(f"  - FNR:            {fnr:.6f} ({fn}/{fn+tp})")
    print("-"*50)
    print("CONFUSION MATRIX:")
    print(f"  True Negative (TN):  {tn}")
    print(f"  False Positive (FP): {fp}")
    print(f"  False Negative (FN): {fn}")
    print(f"  True Positive (TP):  {tp}")
    print("\nVisual representation:")
    print(f"           Predicted 0  Predicted 1")
    print(f"Actual 0   {tn:<11d}  {fp:<11d}")
    print(f"Actual 1   {fn:<11d}  {tp:<11d}")
    print("="*50 + "\n")
    
    # 8. Save trained model weights
    model_file = "advanced_lstm_model.pt"
    print(f"Saving trained model weights to {model_file}...")
    torch.save(model.state_dict(), model_file)
    print("Model weights saved successfully.")

if __name__ == "__main__":
    main()
