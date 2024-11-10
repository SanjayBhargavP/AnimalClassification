import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from networkx.algorithms.bipartite.basic import color

file_name = sys.argv[1]
output_path = sys.argv[2]

os.makedirs(output_path, exist_ok=True)

df = pd.read_csv(file_name)

# Define larger font and figure size
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (10, 6)

metrics = ["Validation Loss", "Validation Accuracy", "Precision", "Recall", "F1 Score"]
colors = ['royalblue', 'darkorange', 'green', 'purple', 'red']

for metric, color  in zip(metrics, colors):
    plt.figure()
    plt.plot(df["Epoch"], df[metric], linestyle='-', linewidth=2 ,color=color)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust save path for readability and structure
    plt.tight_layout()
    plt.savefig(f"{output_path}/{metric.replace(' ', '_')}_vs_Epoch.png", dpi=300)
    plt.close()
