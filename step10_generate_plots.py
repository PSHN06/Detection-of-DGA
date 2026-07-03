import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set seaborn style for clean, academic plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15
})

# 1. Performance Data
# Metrics: Accuracy, F1-Score, FPR, FNR
# Models: LightGBM, XGBoost, LSTM
# Datasets: Hold-Out Test Set (Dataset A), Dataset B External Generalization
data_perf = {
    'Model': [
        'LightGBM', 'LightGBM',
        'XGBoost', 'XGBoost',
        'LSTM (Sequence)', 'LSTM (Sequence)'
    ],
    'Dataset': [
        'Hold-Out Test Set', 'Dataset B External',
        'Hold-Out Test Set', 'Dataset B External',
        'Hold-Out Test Set', 'Dataset B External'
    ],
    'Accuracy': [0.990499, 0.737076, 0.989852, 0.733120, 0.988623, 0.835648],
    'F1-Score': [0.978746, 0.649013, 0.977309, 0.642040, 0.974538, 0.806473],
    'FPR': [0.010962, 0.012022, 0.011537, 0.012439, 0.012069, 0.013599],
    'FNR': [0.004309, 0.513826, 0.005217, 0.521321, 0.008921, 0.315106]
}

df_perf = pd.DataFrame(data_perf)

# Plot 1: Performance Comparison Grouped Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = [('Accuracy', 'Accuracy'), ('F1-Score', 'F1-Score'), ('FPR', 'False Positive Rate (FPR)'), ('FNR', 'False Negative Rate (FNR)')]

# Color palette: Hold-Out Test Set vs. Dataset B External Generalization Set (professional palette)
palette = {'Hold-Out Test Set': '#4C72B0', 'Dataset B External': '#C44E52'}

for idx, (col, name) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    sns.barplot(
        data=df_perf,
        x='Model',
        y=col,
        hue='Dataset',
        palette=palette,
        ax=ax,
        edgecolor='0.2'
    )
    ax.set_title(f'Model Comparison: {name}')
    ax.set_xlabel('Classification Model')
    ax.set_ylabel(name)
    ax.legend(title='Evaluation Dataset')
    
    # Format axes labels for rate/ratios
    if col in ['Accuracy', 'F1-Score']:
        ax.set_ylim(0, 1.05)
    elif col == 'FPR':
        ax.set_ylim(0, 0.025)
    elif col == 'FNR':
        ax.set_ylim(0, 0.65)

plt.suptitle("Model Evaluation: Hold-Out Test Set (Dataset A) vs. Blind Generalization (Dataset B)", y=0.98)
plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300)
plt.close()

# 2. Computational Trade-offs Data
data_comp = {
    'Model': ['LightGBM', 'XGBoost', 'LSTM (Sequence)'],
    'Training Time (s)': [5.9709, 3.8787, 80.8384],
    'Inference Latency (µs/sample)': [1.0988, 2.0643, 25.5621]
}

df_comp = pd.DataFrame(data_comp)

# Plot 2: Computational Trade-offs (Training Time vs Inference Latency)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Training Time
sns.barplot(
    data=df_comp,
    x='Model',
    y='Training Time (s)',
    hue='Model',
    palette='muted',
    ax=axes[0],
    edgecolor='0.2',
    legend=False
)
axes[0].set_title('Training Time Comparison')
axes[0].set_xlabel('Classification Model')
axes[0].set_ylabel('Training Time (seconds) - Log Scale')
axes[0].set_yscale('log') # Log scale because LSTM is much slower to train on CPU
# Annotate values
for p in axes[0].patches:
    val = p.get_height()
    if val > 0:
        axes[0].annotate(f"{val:.2f}s", (p.get_x() + p.get_width() / 2., val),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 2),
                    textcoords='offset points')

# Right plot: Inference Latency
sns.barplot(
    data=df_comp,
    x='Model',
    y='Inference Latency (µs/sample)',
    hue='Model',
    palette='muted',
    ax=axes[1],
    edgecolor='0.2',
    legend=False
)
axes[1].set_title('Inference Latency per Sample')
axes[1].set_xlabel('Classification Model')
axes[1].set_ylabel('Latency (microseconds / sample) - Log Scale')
axes[1].set_yscale('log') # Log scale because LSTM is much slower to run on CPU
# Annotate values
for p in axes[1].patches:
    val = p.get_height()
    if val > 0:
        axes[1].annotate(f"{val:.2f} µs", (p.get_x() + p.get_width() / 2., val),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 2),
                    textcoords='offset points')

plt.suptitle("Computational Efficiency & Latency Trade-offs", y=0.98)
plt.tight_layout()
plt.savefig('computational_tradeoffs.png', dpi=300)
plt.close()

print("Plots generated successfully and saved as model_performance_comparison.png and computational_tradeoffs.png.")
