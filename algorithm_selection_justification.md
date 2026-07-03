# Technical Justification for Algorithm Selection

This document provides the formal technical justifications explaining why our selected machine learning algorithms are suitable for DGA (Domain Generation Algorithm) detection, directly satisfying **Section 6 of the Project Guidelines** ("Random selection of algorithms is not acceptable").

---

## 1. Baseline Model: LightGBM (Light Gradient Boosting Machine)

* **Role**: Baseline Classical Classifier.
* **Why it was selected**:
  * **Gradient-based One-Side Sampling (GOSS)**: DGA detection involves large datasets (over 210,000 training samples). LightGBM keeps data instances with large gradients and runs random sampling on instances with small gradients, allowing it to train much faster than standard tree algorithms while retaining accuracy.
  * **Leaf-wise (Best-First) Tree Growth**: LightGBM grows trees leaf-wise rather than level-wise. This allows it to capture complex, non-linear character interaction splits much deeper in the tree, which is crucial for identifying complex character distributions (like high entropy or digit sequences).
  * **Feature Sparsity Handling (Exclusive Feature Bundling)**: Our 500 n-gram columns are extremely sparse. LightGBM bundles exclusive sparse features together, reducing feature dimensions and accelerating execution, making it a highly efficient baseline.

---

## 2. Second Model: XGBoost (eXtreme Gradient Boosting)

* **Role**: Advanced Tabular Classifier.
* **Why it was selected**:
  * **Sequential Error Correction (Boosting)**: XGBoost trains trees sequentially to correct the misclassifications of preceding trees. This helps the system learn borderline DGA domains that closely mimic natural language (which LightGBM might split crudely).
  * **Cost-Sensitive Loss Optimization**: XGBoost allows configuration of the `scale_pos_weight` hyperparameter to penalize missed DGA detections (False Negatives), minimizing the False Negative Rate (FNR) on live traffic.
  * **Regularization Bounds**: XGBoost includes L1 and L2 regularization to penalize leaf weights, preventing overfitting on specific domain templates.

---

## 3. Third Model: Recurrent Neural Network (LSTM / GRU)

* **Role**: Sequential Deep Learning Model.
* **Why it was selected**:
  * **Captures Character Sequences Natively**: While trees process character counts (n-grams) independently, LSTMs evaluate the domains as a sequence over time (retaining character ordering). This helps distinguish natural transitions from random consonant generation logic.
  * **Dense Embedding Mappings**: The model maps character indices to dense vectors (`nn.Embedding`), clustering similar symbols together (e.g. vowels vs rare DGA consonants).
  * **Latency Adaptations**: Shortening sequence windows from 64 to 32 characters removes zero-padding calculations, optimizing it for real-time CPU deployments.

---

## 4. Meta-Classifier: Hybrid Stacking Ensemble

* **Role**: Final Stacking Ensemble.
* **Why it was selected**:
  * **Feature Fusion**: It stacks the soft sequential DGA confidence of the LSTM as a 505th feature column for the gradient boosted trees, achieving sequence-level accuracy at boosting tree inference speeds.
