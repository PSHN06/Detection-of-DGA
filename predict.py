import pandas as pd
import numpy as np
import math
import sys
import os
from collections import Counter
from scipy.sparse import hstack, csr_matrix
import joblib

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
    """Extract lexical features from a series of domain names."""
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

def clean_domain(domain_or_host):
    """
    Extract the domain name from a host.
    If the user enters a full host (e.g., 'google.com'), we extract the SLD ('google').
    However, if they enter a domain without suffix, we use it directly.
    We split on '.' and handle some basic cases.
    """
    domain_or_host = domain_or_host.strip().lower()
    if not domain_or_host:
        return ""
    
    # Simple extraction of the main domain label:
    # E.g. 'google.com' -> 'google'
    # 'google.co.uk' -> 'google'
    # 'subdomain.google.com' -> 'google'
    parts = domain_or_host.split('.')
    if len(parts) == 1:
        return parts[0]
    
    # If the last parts are TLDs (e.g., co.uk, com, org), extract the SLD
    # A simple suffix check:
    common_suffixes = {'com', 'net', 'org', 'info', 'biz', 'us', 'uk', 'ru', 'cn', 'de', 'jp', 'fr', 'it', 'nl', 'se', 'no', 'es', 'mil', 'gov', 'edu'}
    
    # Traverse from right to left
    idx = len(parts) - 1
    while idx >= 0:
        if parts[idx] in common_suffixes or len(parts[idx]) <= 3:
            idx -= 1
        else:
            break
            
    if idx >= 0:
        return parts[idx]
    else:
        # Fallback to the second to last part if possible, or just the first part
        return parts[-2] if len(parts) >= 2 else parts[0]

def main():
    # File paths
    vectorizer_path = 'tfidf_vectorizer.joblib'
    binary_model_path = 'dga_binary_model.joblib'
    multiclass_model_path = 'dga_multiclass_model.joblib'
    
    # Verify model files exist
    missing_files = [f for f in [vectorizer_path, binary_model_path, multiclass_model_path] if not os.path.exists(f)]
    if missing_files:
        print("Error: The following model files are missing:")
        for f in missing_files:
            print(f" - {f}")
        print("\nPlease run 'train_model.py' first to train and save the models.")
        return

    print("Loading models and vectorizer...")
    vectorizer = joblib.load(vectorizer_path)
    binary_model = joblib.load(binary_model_path)
    multiclass_model = joblib.load(multiclass_model_path)
    print("Models loaded successfully!")

    # Check for CLI arguments
    if len(sys.argv) > 1:
        domains_to_test = sys.argv[1:]
    else:
        print("\n=== DGA Predictor CLI ===")
        print("Enter a domain name to test (or multiple separated by space).")
        print("Press Enter without input to exit.")
        user_input = input("Domain(s): ").strip()
        if not user_input:
            print("Exiting...")
            return
        domains_to_test = user_input.split()

    # Preprocess inputs
    cleaned_domains = [clean_domain(d) for d in domains_to_test]
    
    print(f"\nProcessing {len(domains_to_test)} inputs:")
    for orig, clean in zip(domains_to_test, cleaned_domains):
        print(f" - '{orig}' -> extracted domain: '{clean}'")

    # Extract features
    lexical_feats = extract_lexical_features(cleaned_domains)
    vec_feats = vectorizer.transform(cleaned_domains)
    combined_feats = hstack([vec_feats, csr_matrix(lexical_feats)])

    # Predict
    preds_bin = binary_model.predict(combined_feats)
    probs_bin = binary_model.predict_proba(combined_feats)[:, 1]
    preds_mul = multiclass_model.predict(combined_feats)

    # Print results
    print("\n" + "="*80)
    print(f"{'Input Host/Domain':<30} | {'Extracted':<15} | {'DGA?':<8} | {'DGA Prob':<10} | {'Predicted Subclass':<15}")
    print("="*80)
    for orig, clean, pred_bin, prob_bin, pred_mul in zip(domains_to_test, cleaned_domains, preds_bin, probs_bin, preds_mul):
        is_dga_str = "DGA" if pred_bin == 1 else "Legit"
        prob_str = f"{prob_bin * 100:.1f}%"
        print(f"{orig:<30} | {clean:<15} | {is_dga_str:<8} | {prob_str:<10} | {pred_mul:<15}")
    print("="*80)

if __name__ == '__main__':
    main()
