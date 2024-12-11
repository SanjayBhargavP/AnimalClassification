import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from networkx.algorithms.bipartite.basic import color

file_name = sys.argv[1]
output_path = sys.argv[2]

os.makedirs(output_path, exist_ok=True)

df = pd.read_csv(file_name)

# normalize validation accuracy
df["Validation Accuracy"] = df["Validation Accuracy"] / 100

# Filter out epochs greater than 100
df = df[df["Epoch"] <= 100]

# Define larger font and figure size
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Validation Loss"], linestyle='-', linewidth=2, color='royalblue', label="Validation Loss")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Validation Loss", fontsize=14)
plt.title("Validation Loss vs. Epoch", fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
output_file_loss = f"{output_path}/Validation_Loss_vs_Epoch.png"
plt.savefig(output_file_loss, dpi=300)
print(f"Validation Loss plot saved to {output_file_loss}")

# Plot other metrics together
metrics = ["Validation Accuracy", "Precision", "Recall", "F1 Score"]
colors = ['darkorange', 'green', 'purple', 'red']

plt.figure(figsize=(10, 6))
for metric, color in zip(metrics, colors):
    plt.plot(df["Epoch"], df[metric], linestyle='-', linewidth=2, color=color, label=metric)

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Metric Value", fontsize=14)
plt.title("Metrics vs. Epoch", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
output_file_metrics = f"{output_path}/Metrics_Excluding_Validation_Loss_vs_Epoch.png"
plt.savefig(output_file_metrics, dpi=300)
print(f"Other metrics plot saved to {output_file_metrics}")