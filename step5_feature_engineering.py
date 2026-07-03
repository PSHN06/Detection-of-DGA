import os
import math
import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

def shannon_entropy(s):
    if not s or not isinstance(s, str):
        return 0.0
    counts = Counter(s)
    entropy = 0.0
    total = len(s)
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def digit_density(s):
    if not s or not isinstance(s, str):
        return 0.0
    digit_count = sum(1 for c in s if c.isdigit())
    return digit_count / len(s)

def vowel_to_consonant_ratio(s):
    if not s or not isinstance(s, str):
        return 0.0
    s_lower = s.lower()
    vowels = set('aeiou')
    consonants = set('bcdfghjklmnpqrstvwxyz')
    num_vowels = sum(1 for c in s_lower if c in vowels)
    num_consonants = sum(1 for c in s_lower if c in consonants)
    return float(num_vowels) / (num_consonants + 1)

def get_sld(d):
    d_str = str(d).lower().strip()
    if d_str.startswith('www.'):
        d_str = d_str[4:]
    return d_str.split('.')[0]

def extract_features(df):
    lengths = df['sld'].apply(len)
    entropies = df['sld'].apply(shannon_entropy)
    densities = df['sld'].apply(digit_density)
    ratios = df['sld'].apply(vowel_to_consonant_ratio)
    
    features_df = pd.DataFrame({
        'length': lengths,
        'entropy': entropies,
        'digit_density': densities,
        'vowel_consonant_ratio': ratios
    })
    return features_df

def main():
    # 1. Independently load all datasets
    print("Loading datasets...")
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    b_df = pd.read_csv("dataset_b_external.csv")
    
    # Handle possible NaN values in domains
    train_df['domain'] = train_df['domain'].fillna('')
    val_df['domain'] = val_df['domain'].fillna('')
    test_df['domain'] = test_df['domain'].fillna('')
    b_df['domain'] = b_df['domain'].fillna('')
    
    # Preprocess to get SLDs
    train_df['sld'] = train_df['domain'].apply(get_sld)
    val_df['sld'] = val_df['domain'].apply(get_sld)
    test_df['sld'] = test_df['domain'].apply(get_sld)
    b_df['sld'] = b_df['domain'].apply(get_sld)
    
    print(f"Loaded datasets:")
    print(f"  - Train: {train_df.shape[0]} rows")
    print(f"  - Val: {val_df.shape[0]} rows")
    print(f"  - Test: {test_df.shape[0]} rows")
    print(f"  - Dataset B: {b_df.shape[0]} rows")
    
    # 2. Extract lexical features
    print("Extracting lexical features...")
    X_train_lex = extract_features(train_df)
    X_val_lex = extract_features(val_df)
    X_test_lex = extract_features(test_df)
    X_b_lex = extract_features(b_df)
    
    # 3. Fit CountVectorizer for n-grams on train SLDs ONLY
    print("Fitting CountVectorizer (n-grams 2-3, max 500 features) on train SLDs...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3), max_features=500)
    vectorizer.fit(train_df['sld'])
    
    # Save CountVectorizer checkpoint
    joblib.dump(vectorizer, "ngram_vectorizer.pkl")
    print("CountVectorizer saved to ngram_vectorizer.pkl.")
    
    # Transform domains to n-gram matrices
    print("Transforming all datasets to structural n-grams...")
    train_ngrams = vectorizer.transform(train_df['sld']).toarray()
    val_ngrams = vectorizer.transform(val_df['sld']).toarray()
    test_ngrams = vectorizer.transform(test_df['sld']).toarray()
    b_ngrams = vectorizer.transform(b_df['sld']).toarray()
    
    # Create DataFrames for n-grams
    ngram_cols = [f"ngram_{i}" for i in range(train_ngrams.shape[1])]
    X_train_ngrams = pd.DataFrame(train_ngrams, columns=ngram_cols)
    X_val_ngrams = pd.DataFrame(val_ngrams, columns=ngram_cols)
    X_test_ngrams = pd.DataFrame(test_ngrams, columns=ngram_cols)
    X_b_ngrams = pd.DataFrame(b_ngrams, columns=ngram_cols)
    
    # Concatenate lexical features with n-grams
    print("Concatenating lexical and structural features...")
    X_train_combined = pd.concat([X_train_lex, X_train_ngrams], axis=1)
    X_val_combined = pd.concat([X_val_lex, X_val_ngrams], axis=1)
    X_test_combined = pd.concat([X_test_lex, X_test_ngrams], axis=1)
    X_b_combined = pd.concat([X_b_lex, X_b_ngrams], axis=1)
    
    # 4. Fit StandardScaler ONLY on combined train features
    print("Fitting StandardScaler exclusively on combined train features...")
    scaler = StandardScaler()
    scaler.fit(X_train_combined)
    
    # Save StandardScaler checkpoint
    joblib.dump(scaler, "feature_scaler.pkl")
    print("StandardScaler saved to feature_scaler.pkl.")
    
    # Transform
    print("Scaling features...")
    X_train_scaled = scaler.transform(X_train_combined)
    X_val_scaled = scaler.transform(X_val_combined)
    X_test_scaled = scaler.transform(X_test_combined)
    X_b_scaled = scaler.transform(X_b_combined)
    
    # Convert back to DataFrames with columns
    combined_cols = list(X_train_combined.columns)
    X_train_df = pd.DataFrame(X_train_scaled, columns=combined_cols)
    X_val_df = pd.DataFrame(X_val_scaled, columns=combined_cols)
    X_test_df = pd.DataFrame(X_test_scaled, columns=combined_cols)
    X_b_df = pd.DataFrame(X_b_scaled, columns=combined_cols)
    
    # 5. Save outputs
    print("Saving processed arrays...")
    X_train_df.to_csv("X_train.csv", index=False)
    X_val_df.to_csv("X_val.csv", index=False)
    X_test_df.to_csv("X_test.csv", index=False)
    X_b_df.to_csv("X_dataset_b.csv", index=False)
    
    train_df[['class']].to_csv("y_train.csv", index=False)
    val_df[['class']].to_csv("y_val.csv", index=False)
    test_df[['class']].to_csv("y_test.csv", index=False)
    b_df[['class']].to_csv("y_dataset_b.csv", index=False)
    
    print("Feature engineering and scaling pipeline finished successfully.")

if __name__ == "__main__":
    main()
