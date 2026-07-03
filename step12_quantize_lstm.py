import os
import torch
import json
from step8_train_lstm import LSTMClassifier

def main():
    # 1. Load baseline checkpoint
    print("Loading baseline LSTM model...")
    with open("lstm_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    base_model = LSTMClassifier(vocab_size=vocab_size)
    base_model.load_state_dict(torch.load("advanced_lstm_model.pt", map_location=torch.device('cpu')))
    base_model.eval()
    
    # 2. Dynamic Quantization
    print("Executing dynamic quantization (Float32 -> Int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.LSTM, torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # 3. Export quantized checkpoint
    quantized_path = "lstm_quantized.pt"
    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"Quantized model saved to {quantized_path}")
    
    # 4. Measure file sizes
    orig_path = "advanced_lstm_model.pt"
    orig_size = os.path.getsize(orig_path) / (1024 * 1024) # size in MB
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024) # size in MB
    
    print("\n" + "="*50)
    print("             QUANTIZATION METRICS SUMMARY            ")
    print("="*50)
    print(f"Original Model Size:  {orig_size:.6f} MB ({os.path.getsize(orig_path)} bytes)")
    print(f"Quantized Model Size: {quant_size:.6f} MB ({os.path.getsize(quantized_path)} bytes)")
    print(f"Size Reduction Ratio: {(1.0 - quant_size/orig_size)*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
