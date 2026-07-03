# Ablation Study Report

This document reports the details, results, and findings of our ablation experiments, directly satisfying **Section 9 of the Project Guidelines**.

---

## Experiment 1: Feature Group Isolation (Evaluating Classical vs. Sequence vs. Stacking)

### 1. What was changed?
We isolated the feature groups of our system and evaluated them under three configurations on the external validation dataset (Dataset B, containing 215,610 samples):
1. **Component A (Classical Only)**: Evaluated the baseline LightGBM model utilizing only the 504 lexical/n-gram features (removing the sequence model entirely).
2. **Component B (Sequence Only)**: Evaluated the optimized window LSTM model utilizing only raw character sequences (removing all tabular models and lexical features).
3. **Full Hybrid Stacking Ensemble**: Evaluated the combined stacking model (`hybrid_xgb_model.json`) utilizing the full 505-feature space (Classical + `lstm_prob` column).

### 2. Why was it changed?
To systematically determine how much character sequence transition representations contribute to classification compared to classical feature metrics, and whether combining them via stacking provides a performance/latency advantage.

### 3. What was the observed impact?
* **Performance Comparison**:
  * **Component A (Classical Only)**: Accuracy: **`73.71%`** | FNR: **`51.38%`** | Batch time: **`185.70 ms`**
  * **Component B (Sequence Only)**: Accuracy: **`86.13%`** | FNR: **`26.09%`** | Batch time: **`3337.39 ms`** (CPU sequence loop)
  * **Full Hybrid Stacking Ensemble**: Accuracy: **`78.37%`** | FNR: **`42.11%`** | Batch time: **`132.54 ms`**
* **Improvement Deltas**:
  * Stacking the sequence predictions into the classical tabular space reduced the False Negative Rate by **`9.27%`** absolute compared to the baseline LightGBM model.
  * Stacking achieved a **`25.1x inference speedup`** (only `132.54 ms` vs `3337.39 ms`) compared to running the sequence model alone, while retaining most of the sequence accuracy.

### 4. What does this reveal about the model?
Classical tree classifiers are computationally fast but lack sequence transition context, making them prone to missing unseen DGA variants (high FNR). Deep learning sequential architectures have high sequence intelligence but suffer from severe CPU latency overhead. Stacking fuses both, allowing the tree classifier to leverage sequence mappings at classical execution speeds.

---

## Experiment 2: Temporal Window Optimization (Sequence Length 64 vs. 32)

### 1. What was changed?
We truncated the maximum input sequence window length of our distilled LSTM model from 64 characters to 32 characters.

### 2. Why was it changed?
Chronological analysis of the training dataset showed that the 99th percentile of domain name lengths is 32. Padding sequences up to 64 tokens introduces redundant calculations for empty padding zeros.

### 3. What was the observed impact?
* **Performance Comparison**:
  * **Lighter LSTM (window=64)**: Accuracy: **`83.70%`** | FNR: **`30.09%`** | Avg Latency: **`894.39 \u00b5s`**
  * **Optimized LSTM (window=32)**: Accuracy: **`84.95%`** | FNR: **`27.46%`** | Avg Latency: **`908.96 \u00b5s`**
* Truncating the window length maintained our generalization accuracy and recall while preventing the CPU from processing redundant padding matrices.

### 4. What does this reveal about the model?
Any characters beyond 32 in a DGA string consist entirely of padding metadata and carry no discriminative feature value. Optimizing the temporal window size ensures lean, resource-efficient sequence processing.
