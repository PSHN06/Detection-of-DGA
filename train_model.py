import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import joblib
import time
import os

def calculate_entropy(s):
    """Calculate Shannon entropy of a string."""
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    cnt = Counter(s)
    entropy = 0.0
    for val in cnt.values():
        p = val / len(s)
        entropy -= p * math.log2(p)
    return entropy

def extract_lexical_features(series):
    """
    Extract lexical features from a series of domain names.
    Features:
    1. Length of domain
    2. Shannon entropy
    3. Digit ratio (number of digits / length)
    4. Vowel ratio (number of vowels / length)
    5. Consonant ratio (number of consonants / length)
    6. Non-alphanumeric ratio (e.g., hyphens / length)
    """
    features = []
    for s in series:
        if not isinstance(s, str):
            s = ""
        length = len(s)
        entropy = calculate_entropy(s)
        
        digits = sum(c.isdigit() for c in s)
        digit_ratio = digits / length if length > 0 else 0.0
        
        vowels = sum(c.lower() in 'aeiou' for c in s)
        vowel_ratio = vowels / length if length > 0 else 0.0
        
        consonants = sum(c.lower() in 'bcdfghjklmnpqrstvwxyz' for c in s)
        consonant_ratio = consonants / length if length > 0 else 0.0
        
        non_alnum = sum(not c.isalnum() for c in s)
        non_alnum_ratio = non_alnum / length if length > 0 else 0.0
        
        features.append([length, entropy, digit_ratio, vowel_ratio, consonant_ratio, non_alnum_ratio])
    return np.array(features)

def main():
    data_path = "dga_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found in the current directory.")
        return

    print("Step 1: Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Preprocessing
    print(f"Original shape: {df.shape}")
    df = df.dropna(subset=['domain'])
    print(f"Shape after dropping null domains: {df.shape}")

    # Prepare features and targets
    X_domain = df['domain']
    y_binary = (df['isDGA'] == 'dga').astype(int).values
    y_multi = df['subclass'].values

    # Train/test split (80/20) - we use stratify on subclass to keep same distributions
    print("\nStep 2: Splitting dataset into train and test sets...")
    X_train_dom, X_test_dom, y_train_bin, y_test_bin, y_train_mul, y_test_mul = train_test_split(
        X_domain, y_binary, y_multi, test_size=0.2, stratify=y_multi, random_state=42
    )

    print(f"Train samples: {len(X_train_dom)}, Test samples: {len(X_test_dom)}")

    # Feature extraction
    print("\nStep 3: Extracting lexical features...")
    start = time.time()
    X_train_lex = extract_lexical_features(X_train_dom)
    X_test_lex = extract_lexical_features(X_test_dom)
    print(f"Lexical feature extraction completed in {time.time() - start:.2f} seconds.")

    print("\nStep 4: Extracting TF-IDF char n-gram features (2 to 4 ngrams)...")
    start = time.time()
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=20000)
    X_train_vec = vectorizer.fit_transform(X_train_dom)
    X_test_vec = vectorizer.transform(X_test_dom)
    print(f"TF-IDF Vectorization completed in {time.time() - start:.2f} seconds.")
    print(f"TF-IDF vocabulary size: {X_train_vec.shape[1]}")

    # Combine features
    print("\nStep 5: Combining features...")
    X_train_comb = hstack([X_train_vec, csr_matrix(X_train_lex)])
    X_test_comb = hstack([X_test_vec, csr_matrix(X_test_lex)])
    print(f"Combined feature shape (train): {X_train_comb.shape}")
    print(f"Combined feature shape (test): {X_test_comb.shape}")

    # Train Binary Classifier
    print("\nStep 6: Training Binary Random Forest Classifier...")
    start = time.time()
    rf_binary = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)
    rf_binary.fit(X_train_comb, y_train_bin)
    print(f"Binary model trained in {time.time() - start:.2f} seconds.")

    # Evaluate Binary Classifier
    print("\nEvaluating Binary Classifier...")
    y_pred_bin = rf_binary.predict(X_test_comb)
    y_prob_bin = rf_binary.predict_proba(X_test_comb)[:, 1]
    
    bin_accuracy = accuracy_score(y_test_bin, y_pred_bin)
    bin_roc_auc = roc_auc_score(y_test_bin, y_prob_bin)
    
    print(f"Binary Classifier Accuracy: {bin_accuracy * 100:.2f}%")
    print(f"Binary Classifier ROC AUC: {bin_roc_auc:.4f}")
    print("\nClassification Report (Binary):")
    print(classification_report(y_test_bin, y_pred_bin, target_names=['legit', 'dga']))

    # Train Multi-Class Classifier
    print("\nStep 7: Training Multi-Class Random Forest Classifier (Subclass prediction)...")
    start = time.time()
    rf_multi = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)
    rf_multi.fit(X_train_comb, y_train_mul)
    print(f"Multi-class model trained in {time.time() - start:.2f} seconds.")

    # Evaluate Multi-Class Classifier
    print("\nEvaluating Multi-Class Classifier...")
    y_pred_mul = rf_multi.predict(X_test_comb)
    
    mul_accuracy = accuracy_score(y_test_mul, y_pred_mul)
    print(f"Multi-class Classifier Accuracy: {mul_accuracy * 100:.2f}%")
    print("\nClassification Report (Multi-Class):")
    # Get unique subclass names in sorted order
    classes = sorted(list(np.unique(y_multi)))
    print(classification_report(y_test_mul, y_pred_mul, target_names=classes))

    # Saving models and vectorizer
    print("\nStep 8: Saving models and vectorizer to disk...")
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(rf_binary, 'dga_binary_model.joblib')
    joblib.dump(rf_multi, 'dga_multiclass_model.joblib')
    print("Saved files successfully:")
    print(" - tfidf_vectorizer.joblib")
    print(" - dga_binary_model.joblib")
    print(" - dga_multiclass_model.joblib")

if __name__ == '__main__':
    main()
