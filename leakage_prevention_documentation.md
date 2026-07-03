# Information Leakage Prevention Documentation

This document describes the design patterns and methodological precautions implemented to eliminate information leakage, as required by **Section 5 of the Project Guidelines**.

---

## 1. Temporal-Aware Separation (Section 4 Compliance)

One of the most common mistakes in DGA detection research is random shuffling before train-test splitting. Because DGA seeds and benign registrations evolve chronologically, random shuffling leaks future domain structures into the training partition.

### Leakage Prevention Measure:
* **Chronological Sorting**: The raw master dataset is sorted strictly by `timestamp` from the oldest observation to the newest before partition division.
* **Non-Shuffled Splits**:
  * **Oldest 70%** (Train partition): Allocated to train the CountVectorizer, StandardScaler, and neural embeddings.
  * **Middle 10%** (Validation partition): Reserved for hyperparameter tuning.
  * **Newest 20%** (Testing partition): Reserved for hold-out performance benchmarking.
* **Temporal Validity**: This guarantees that validation and testing always represent future unseen observations relative to the training period, simulating a real-world deployment.

---

## 2. Fit Bounds & Pipeline Isolation (Section 5 Compliance)

Feature engineering and data normalization can easily leak data properties from the test partition into the training partition if computed on the entire dataset.

### Leakage Prevention Measures:
1. **No Complete Dataset Normalization**:
   * The `StandardScaler` checkpoint (`feature_scaler.pkl`) is fitted **exclusively** on the training partition (`X_train`). It is subsequently used to transform validation (`X_val`), testing (`X_test`), and external evaluation (`X_b`) matrices.
2. **Pre-split Feature Vocabulary Mapping**:
   * The character n-gram vocabulary is fitted using `CountVectorizer` **exclusively** on the training partition SLDs (`train.csv`). The validation and testing splits are transformed strictly using this pre-fitted vocabulary map. Any n-gram occurring in validation or testing that was not present in the training set is ignored.
3. **No Target Leakage in Sequence Processing**:
   * Character-to-integer vocabularies are mapped from `lstm_vocab.json` built solely on the training partition characters.

---

## 3. Duplicate Audits

Including identical domain names across different splits allows the model to memorize labels, falsely boosting test scores.

### Leakage Prevention Measure:
* **Deduplication Audit**: We executed an explicit deduplication script (`step2.5_deduplicate_dataset.py`) to drop duplicate domain names within both the malicious and benign raw datasets. Over **167,000 duplicate domains** were purged prior to splitting, ensuring that the test set evaluates the model's capacity to classify *new, structurally distinct* domains.
