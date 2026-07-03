# Generalization Study Report (Dataset B Evaluation)

A major limitation of cybersecurity machine learning models is overfitting to known lab datasets. To measure our models' true robustness, we evaluated them on **Dataset B External Generalization Set** (215,610 samples) containing completely unseen malware families. No model was retrained for this study.

---

## 1. Generalization Performance Summary

| Model Configuration | Accuracy (Dataset A) | Accuracy (Dataset B) | FNR (Dataset A) | FNR (Dataset B) | FNR Degradation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline LightGBM** | 99.0499% | 73.7076% | 0.4309% | 51.3826% | **+50.95% (FNR)** |
| **Advanced XGBoost** | 98.9852% | 73.3120% | 0.5217% | 52.1321% | **+51.61% (FNR)** |
| **Baseline LSTM** | 98.8623% | 83.5648% | 0.8921% | 31.5106% | **+30.62% (FNR)** |
| **Hybrid Stacking LGB** | 99.1164% | 77.2135% | 0.4839% | 44.4979% | **+44.01% (FNR)** |
| **Hybrid Stacking XGB** | 99.0499% | 78.3702% | 0.5519% | 42.1121% | **+41.56% (FNR)** |

---

## 2. Analysis of Performance Degradation

When evaluated on Dataset B, all models experienced performance drops, but the baseline tree classifiers suffered severe degradation—with their **False Negative Rate (FNR) jumping to over 51%**. 

### Why Classical Models Failed on Dataset B:
* **Over-fitting to Fixed Vocabularies**: Classical classifiers rely heavily on character n-gram counts (e.g. bigrams/trigrams). Because Dataset B contains unseen malware families, these new families use completely different character distribution seeds. The n-gram frequencies learned during training do not exist in the new domains, causing the tree splits to fail.
* **Lexical Feature Mimicry**: Vowel-consonant ratios and entropy scores can be easily bypassed by dictionary-based DGAs that concatenate real English words to construct domains.

### Why Sequence Models Remained Robust:
* **Dynamic Transition Mapping**: The character-level LSTM model maps character sequences step-by-step. Instead of looking for specific letters, it evaluates the general structure of the string. This allows it to identify character-level randomness natively, maintaining a much lower FNR (**31.51%**).

---

## 3. Stacking Ensemble Mitigation
While the pure LSTM is robust, running it on CPU-based network gateways is slow. 

By using the **Hybrid Stacking Ensemble**, we feed the LSTM's DGA probability as a feature into XGBoost. This approach:
* Recovers classical tree accuracy on Dataset B to **78.37%** (a 5.06% absolute increase).
* Reduces the FNR of the baseline tree model by **10.02%** absolute (dropping FNR from 52.13% to 42.11%).
* Restores inference processing times to tabular speeds (132.54 ms per batch vs 3337.39 ms for pure LSTM), making it a viable real-world network gateway defense.
