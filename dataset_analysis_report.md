# Dataset Analysis Report

## 1. Raw Dataset Origins
To build a representative DGA detection model, we collected domains from multiple independent sources to prevent dataset bias:
* **Benign Domain Corpus**: 
  * Source: Tranco Top 1 Million list (a globally aggregated traffic ranking list that filters out active malicious vectors).
  * Characteristics: Features standard natural language structures, readable syllables, and familiar brand terms.
* **Malicious DGA Domain Corpus**:
  * Source: Custom domains generated from reverse-engineered seeds of 5 malware families: `banjori`, `locky`, `necurs`, `pykspa`, and `ramnit`.
  * Characteristics: High entropy, random consonant distributions, and varied lengths based on family-specific rules.

## 2. Duplicate Sample Audit & Data Deduplication
In cybersecurity research, duplicate samples across train and test sets create inflated performance metrics due to data memorization. We ran an audit to clean our raw files:
* **Initial Malicious Rows**: 234,410
* **Initial Benign Rows**: 234,410
* **Duplicate Malicious Domains Dropped**: 167,680
* **Duplicate Benign Domains Dropped**: 107
* **Overlapping Domains (present in both corpora)**: 0

Deduplication significantly reduced the malicious corpus size to **66,730 unique domains**. This prevents test set bias, forcing the models to evaluate new, unseen random structures.

## 3. Class Distribution & Split Strategy
Following deduplication, the total clean dataset contains **301,033 domains**.

* **Class Distribution**:
  * Benign (Class 0): 234,303 samples (77.83%)
  * Malicious (Class 1): 66,730 samples (22.17%)
  * Class Ratio: 3.51 : 1
* **Partition Splits (Chronological Order)**:
  * **Training Set (70%)**: 210,723 samples (163,274 Benign, 47,449 Malicious)
  * **Validation Set (10%)**: 30,103 samples (24,049 Benign, 6,054 Malicious)
  * **Testing Set (20%)**: 60,207 samples (46,980 Benign, 13,227 Malicious)

## 4. Feature Space Schema
The feature space engineered from domain names consists of **504 total features**:
1. **Lexical Metrics (4 features)**:
   * `length`: Character length of the isolated Second-Level Domain (SLD).
   * `entropy`: Shannon entropy score measuring character randomness.
   * `digit_density`: Proportion of numeric digits to total characters.
   * `vowel_consonant_ratio`: Laplace-smoothed ratio of vowels to consonants.
2. **Structural n-grams (500 features)**:
   * Character bigrams and trigrams fitted exclusively on training SLDs (e.g. `co`, `in`, `ing`). Only the top 500 n-grams by frequency are retained.
