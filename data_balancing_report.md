# Class Distribution Analysis & Data Balancing Report

This document reports the class distribution statistics, imbalance severity, and balancing justifications prior to model development, as required by **Section 3 of the Project Guidelines**.

---

## 1. Class Distribution Analysis (Post-Deduplication)

Prior to splitting the dataset, we audited the raw samples to remove duplicates and ensure strict label boundaries.

* **Number of Benign Samples (Class 0)**: 234,303
* **Number of Attack Samples (Class 1)**: 66,730
* **Total Master Samples**: 301,033
* **Class Ratio (Benign : Attack)**: **3.51 : 1**
* **Minority Class Proportion**: **22.17%**
* **Imbalance Severity**: **Mild**

---

## 2. Partition-Level Class Distributions

Following our chronological temporal-aware split (70% train, 10% val, 20% test), the class distributions across partitions are:

| Partition | Benign (Class 0) | Malicious (Class 1) | Total Samples | Minority Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Training (70%)** | 163,274 | 47,449 | 210,723 | 22.52% (Mild) |
| **Validation (10%)** | 24,049 | 6,054 | 30,103 | 20.11% (Mild) |
| **Testing (20%)** | 46,980 | 13,227 | 60,207 | 21.97% (Mild) |

---

## 3. Data Balancing Strategy & Justification

Cybersecurity datasets often exhibit severe class imbalance (e.g. 1000:1 benign to malicious). In our case, the imbalance severity is **Mild (22.52% minority proportion)**.

### Selected Strategy: Cost-Sensitive Loss weighting (No Resampling)
Instead of applying synthetic oversampling (like SMOTE or ADASYN) or undersampling, we elected to train on the natural distribution while utilizing cost-sensitive learning parameters:
1. **XGBoost Class Weighting**:
   * We configured XGBoost using the `scale_pos_weight` hyperparameter set to `3.44` (computed as `num_benign / num_malicious` in the training partition).
2. **PyTorch LSTM Loss Weighting**:
   * We configured the `nn.BCEWithLogitsLoss` using a `pos_weight` tensor to scale the loss of the positive class.

### Justification:
* **Prevention of Overfitting**: Generative oversampling (like SMOTE) on character-level domain structures can introduce synthetic domains with unnatural, unpronounceable n-grams, adding noise and degrading the model's precision.
* **Preservation of Temporal Validity**: Random undersampling would destroy the chronological sequences and temporal density of benign and malicious domain events.
* **Realistic Environment Simulation**: Training with cost-sensitive weights allows our models to learn the true structural features of natural domains while adjusting decision boundaries to penalize false negatives (missed malware) more heavily.
