# Deployment and Execution Guide

This guide provides step-by-step instructions on setting up your Python environment, executing the DGA detection pipeline, and running the live network sniffer demonstration.

---

## 1. Environment Setup

Our pipeline runs on Python 3.8+ (tested on Python 3.13). 

### Install Dependencies:
Open your terminal (PowerShell or Command Prompt) and install the required machine learning and networking libraries:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost scapy torch matplotlib
```

*Note: Real-time network packet sniffing via Scapy on Windows requires **Npcap** or **WinPcap** drivers. If they are not installed, the sniffer will automatically run in simulated fallback mode.*

---

## 2. Pipeline Execution Sequence

If you wish to re-run the entire pipeline from scratch, execute the following steps in order inside the project directory:

### Phase 1: Data Preparation & Preprocessing
1. **Deduplicate Raw Data**: Purges duplicates from raw datasets to prevent leakage.
   ```bash
   python step2.5_deduplicate_dataset.py
   ```
2. **Feature Engineering**: Generates lexical metrics and n-gram indices.
   ```bash
   python step5_feature_engineering.py
   ```

### Phase 2: Model Training
3. **Train LightGBM Baseline**:
   ```bash
   python step6_train_baseline.py
   ```
4. **Train Advanced XGBoost**:
   ```bash
   python step7_train_advanced_xgb.py
   ```
5. **Train Sequence LSTM**:
   ```bash
   python step8_train_lstm.py
   ```
6. **Train Distilled Students**:
   ```bash
   python step13_distill_light_lstm.py
   python step14_distill_gru.py
   python step16_optimize_window_lstm.py
   ```
7. **Train Hybrid Stacking Ensemble**:
   ```bash
   python step19_hybrid_ensemble.py
   ```

---

## 3. Launching the Showdown Dashboard

To run the interactive CLI comparison suite and evaluate individual domain inputs on all 7 models sequentially, run:
```bash
python step20_ultimate_dashboard.py
```
*   **Usage**: Input any domain string (e.g. `google.com` or `vzxqk`) when prompted to view the prediction metrics, latency tracking, and majority vote production verdict.
*   **Demo Mode (Non-interactive)**:
    ```bash
    python step20_ultimate_dashboard.py --demo
    ```

---

## 4. Live DNS Sniffer Demonstration

To demonstrate a live network intrusion detection setup, open **two separate terminal windows**:

### Terminal 1: Launch the Packet Sniffer
The sniffer binds to local network sockets, intercepts UDP port 53 (DNS) queries, and runs the Hybrid Stacking XGBoost model in real-time:
```bash
python step22_traffic_sniffer.py
```

### Terminal 2: Launch the Traffic Emulator
The emulator sends queries for random benign and malicious DGA domains to generate live network events:
```bash
python step23_attack_emulator.py
```

Observe Terminal 1 to watch the sniffer intercept the packets, classify the domains, and display performance telemetry.
