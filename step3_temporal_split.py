import os
import csv

def load_csv(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return data
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def write_csv(file_path, data, fieldnames):
    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def calculate_imbalance_severity(minority_ratio):
    # Severity thresholds based on minority class proportion
    if minority_ratio < 0.01:
        return "Extreme (Anomaly detection level)"
    elif minority_ratio < 0.20:
        return "Moderate"
    elif minority_ratio < 0.40:
        return "Mild"
    else:
        return "None (Balanced)"

def get_stats(data):
    benign_count = sum(1 for row in data if int(row['class']) == 0)
    attack_count = sum(1 for row in data if int(row['class']) == 1)
    total = len(data)
    if total == 0:
        return 0, 0, 0.0
    
    # final class ratio: benign / attack (or vice versa, we print both for clarity)
    # minority class proportion
    minority_ratio = min(benign_count, attack_count) / total if total > 0 else 0.0
    return benign_count, attack_count, minority_ratio

def main():
    malicious_file = "malicious_raw.csv"
    benign_file = "benign_raw.csv"
    
    # 1. Load datasets
    print("Loading datasets...")
    malicious_data = load_csv(malicious_file)
    benign_data = load_csv(benign_file)
    
    print(f"Loaded {len(malicious_data)} malicious rows.")
    print(f"Loaded {len(benign_data)} benign rows.")
    
    # 2. Concatenate
    master_data = malicious_data + benign_data
    total_rows = len(master_data)
    print(f"Combined master dataset has {total_rows} rows.")
    
    if total_rows != 368820:
        print(f"Warning: Expected 368820 rows, but got {total_rows}")
        
    # 3. Sort strictly by timestamp from oldest to newest
    print("Sorting master dataset chronologically by timestamp...")
    master_data.sort(key=lambda x: x['timestamp'])
    
    # 4. Sequentially split (70% train, 10% val, 20% test)
    # Train: 70%, Val: 10%, Test: 20%
    train_idx = round(0.70 * total_rows)
    val_idx = train_idx + round(0.10 * total_rows)
    
    train_partition = master_data[:train_idx]
    val_partition = master_data[train_idx:val_idx]
    test_partition = master_data[val_idx:]
    
    print(f"Split sizes:")
    print(f"  - Train partition: {len(train_partition)} rows")
    print(f"  - Val partition: {len(val_partition)} rows")
    print(f"  - Test partition: {len(test_partition)} rows")
    
    # Save partitions
    fieldnames = ["domain", "timestamp", "family", "class"]
    print("Writing partition CSVs...")
    write_csv("train.csv", train_partition, fieldnames)
    write_csv("val.csv", val_partition, fieldnames)
    write_csv("test.csv", test_partition, fieldnames)
    print("Partitions saved successfully.")
    
    # 5. Calculate statistics and metrics
    # Overall statistics
    total_benign, total_attack, _ = get_stats(master_data)
    
    # Train partition statistics
    train_benign, train_attack, train_minority_ratio = get_stats(train_partition)
    val_benign, val_attack, _ = get_stats(val_partition)
    test_benign, test_attack, _ = get_stats(test_partition)
    
    imbalance_severity = calculate_imbalance_severity(train_minority_ratio)
    
    # Print Summary Report
    print("\n" + "="*50)
    print("                DATASET SUMMARY REPORT                ")
    print("="*50)
    print(f"Total Benign (Class 0) Samples: {total_benign}")
    print(f"Total Attack (Class 1) Samples: {total_attack}")
    
    # Class Ratio
    if total_attack > 0:
        ratio_val = total_benign / total_attack
        print(f"Final Class Ratio (Benign : Attack): {ratio_val:.4f} : 1  ({total_benign}/{total_attack})")
    else:
        print("Final Class Ratio (Benign : Attack): N/A (no attack samples)")
        
    print("-"*50)
    print("TRAINING PARTITION DETAILS:")
    print(f"  - Benign Samples: {train_benign}")
    print(f"  - Attack Samples: {train_attack}")
    print(f"  - Minority Class Ratio: {train_minority_ratio:.4%}")
    print(f"  - Calculated Imbalance Severity: {imbalance_severity}")
    
    print("-"*50)
    print("VALIDATION PARTITION DETAILS:")
    print(f"  - Benign Samples: {val_benign}")
    print(f"  - Attack Samples: {val_attack}")
    
    print("-"*50)
    print("TEST PARTITION DETAILS:")
    print(f"  - Benign Samples: {test_benign}")
    print(f"  - Attack Samples: {test_attack}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
