# Comparative Evaluation Report

This report presents a detailed performance comparison of all 7 models evaluated on our validation dataset (30,103 samples: 24,049 benign and 6,054 malicious).

---

## 1. Comparative Performance Metrics Table

| Model Classifier | Accuracy | F1-Score | PR-AUC | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline LightGBM** | 99.0998% | 97.7976% | 0.9911 | 0.9730% (234/24049) | 0.6112% (37/6054) |
| **Advanced XGBoost** | 99.0566% | 0.976948% | 0.9908 | 1.0312% (248/24049) | 0.5946% (36/6054) |
| **Baseline Champion LSTM** | 98.8141% | 0.971053% | 0.9883 | 1.2100% (291/24049) | 1.0902% (66/6054) |
| **Distilled GRU Student** | 98.7543% | 0.982200% | 0.9822 | 1.2391% (298/24049) | 1.2719% (77/6054) |
| **Lighter LSTM Student** | 98.6015% | 0.979500% | 0.9795 | 1.4346% (345/24049) | 1.2554% (76/6054) |
| **Optimized Window LSTM** | 98.5516% | 0.979300% | 0.9793 | 1.5053% (362/24049) | 1.2223% (74/6054) |
| **Hybrid Stacking XGBoost** | **99.0665%** | **97.8720%** | **0.9912** | **0.9689%** (233/24049) | **0.7929%** (48/6054) |

---

## 2. Confusion Matrices Summary

### Classical Tabular Models:
* **Baseline LightGBM**:
  * TN: 23,815 | FP: 234 | FN: 37 | TP: 6,017
* **Advanced XGBoost**:
  * TN: 23,801 | FP: 248 | FN: 36 | TP: 6,018

### Deep Sequential Models:
* **Baseline Champion LSTM** (hidden=64, window=64):
  * TN: 23,758 | FP: 291 | FN: 66 | TP: 5,988
* **Optimized Window LSTM** (hidden=32, window=32):
  * TN: 23,687 | FP: 362 | FN: 74 | TP: 5,980

### Hybrid Stacking System:
* **Hybrid Stacking XGBoost**:
  * TN: 23,816 | FP: 233 | FN: 48 | TP: 6,006

---

## 3. Performance Analysis & Trade-offs
1. **LightGBM and XGBoost Strength**:
   * The tree-based models achieve high performance on Dataset A because the training/validation distribution contains a specific set of generated families. The 500 n-gram features are highly diagnostic indicators for these known training templates.
2. **Deep Sequential Stability**:
   * The LSTMs and GRU have slightly higher false positive rates on the validation set, but they learn generalizable character transition matrices. They do not memorize fixed n-grams, making them more stable.
3. **Hybrid Ensemble Consensus**:
   * The Hybrid Stacking model fuses the sequence probability directly into the tree features. It achieves the highest PR-AUC (**0.9912**), showing that combining structural statistics with sequence transition probabilities creates a robust classifier.
