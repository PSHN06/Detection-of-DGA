# Dataset Suitability Justification

This document provides a technical justification of the datasets used for the DGA Detection project, outlining their origins, structures, and direct relevance to DGA detection as required by **Section 2 of the Project Guidelines**.

---

## 1. Dataset A: Baseline & Evaluation Dataset
Dataset A serves as the core training, validation, and testing dataset. It represents a combination of benign traffic and generated DGA traffic.

* **Benign Source (Class 0)**:
  * **Origin**: Tranco Top 1 Million list (a highly reliable, community-verified list of top websites ranked by global traffic).
  * **Sample Size**: 234,303 domains (after deduplication).
  * **Label Reliability**: High. Tranco filters out active threat vectors and ephemeral domains, providing a clean baseline of benign natural language domains.
* **Malicious Source (Class 1)**:
  * **Origin**: Custom-generated DGA domain sets emulating active malware families (e.g. Banjori, Locky, Necurs, Pykspa, Ramnit) using their reverse-engineered seed generation logic.
  * **Sample Size**: 66,730 domains (after deduplication).
  * **Label Reliability**: High. Every domain is generated directly from deterministic malware algorithms, ensuring zero label noise or overlap.
* **Relevance of Features**:
  * Character-level structure is the primary discriminator between human-crafted domains (benign) and algorithmically generated domains (malicious). 
  * Features such as domain length, Shannon entropy, digit density, and vowel-consonant ratios directly capture the structural anomalies (like high consonant density or random character distributions) characteristic of DGA domains.

---

## 2. Dataset B: External Generalization Study Dataset
Dataset B is a completely independent dataset used exclusively to evaluate the models' generalization capability on unseen malware families without retraining, as required by **Section 8**.

* **Benign Source (Class 0)**:
  * **Origin**: Majestic Million Top Sites.
  * **Sample Size**: 151,911 domains.
  * **Label Reliability**: High. It acts as an independent benign baseline source distinct from Tranco.
* **Malicious Source (Class 1)**:
  * **Origin**: Netlab 360 DGA feed and independent malware sandbox outputs.
  * **Sample Size**: 63,699 domains.
  * **Label Reliability**: Verified malicious DGA domains collected from real network traffic captures of unseen malware families (families not present in Dataset A).
* **Generalization Suitability**:
  * This dataset contains domain structures from entirely different DGA families, ensuring that the model is tested against out-of-distribution, future unseen network threat variants.
