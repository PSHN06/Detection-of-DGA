import os
import time
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def main():
    # 1. Load preprocessed datasets
    print("Loading preprocessed training and validation sets...")
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    X_val = pd.read_csv("X_val.csv")
    y_val = pd.read_csv("y_val.csv").values.ravel()
    
    print(f"Dataset dimensions:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_val:   {X_val.shape}")
    
    # 2. Initialize LGBMClassifier
    print("Initializing LGBMClassifier...")
    model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    
    # 3. Fit model and track Training Time
    print("Fitting baseline LightGBM model on the training set...")
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    print(f"Training completed successfully.")
    print(f"Training Time: {train_time:.4f} seconds")
    
    # 4. Run inference and measure Inference Latency
    print("Running inference on the validation set...")
    start_inf = time.time()
    y_pred = model.predict(X_val)
    end_inf = time.time()
    y_probs = model.predict_proba(X_val)[:, 1]
    
    total_inf_time = end_inf - start_inf
    latency_per_sample = total_inf_time / len(X_val)
    
    # 5. Calculate validation metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_probs)
    pr_auc = average_precision_score(y_val, y_probs)
    
    # Confusion Matrix metrics
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Print metrics
    print("\n" + "="*50)
    print("           BASELINE LIGHTGBM MODEL EVALUATION REPORT           ")
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
    
    # 6. Save trained model
    model_file = "baseline_lgb_model.pkl"
    print(f"Saving trained model to {model_file}...")
    joblib.dump(model, model_file)
    print("Model saved successfully.")
    
if __name__ == "__main__":
    main()
