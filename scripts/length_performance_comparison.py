import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

full_csv_files = os.listdir("path_to_directory")
all_dfs = [pd.read_csv("path_to_directory" + f"{file}" +  "/formatted.csv") for file in full_csv_files]

# Step 1: Combine all data to find quantile-based bin edges
all_values = pd.concat([df['predicted_evidence_tokens_word_count'] for df in (all_dfs)])
num_bins = 7
_, bin_edges = pd.qcut(all_values, q=num_bins, retbins=True, duplicates='drop') # first qcut gives edges, these edges are used with cut later

# Initialize dictionaries to store recalls
recalls_predicted = {i: [] for i in range(len(bin_edges) - 1)}
recalls_unsupervised = {i: [] for i in range(len(bin_edges) - 1)}
recalls_gt = {i: [] for i in range(len(bin_edges) - 1)}

# Step 2: Process each DataFrame for predicted and ground truth recalls
for df in all_dfs:
    # Bin the predicted evidence word count
    df['predicted_bins'] = pd.cut(
        df['predicted_evidence_tokens_word_count'], bins=bin_edges, include_lowest=True
    )
    predicted_grouped = df.groupby('predicted_bins')['detection'].value_counts().unstack(fill_value=0)
    predicted_grouped['Recall'] = predicted_grouped[0] / (predicted_grouped[1] + predicted_grouped[0])

    # Add predicted recalls to the corresponding bin
    for i, interval in enumerate(bin_edges[:-1]):
        recalls_predicted[i].append(predicted_grouped['Recall'].get(interval, 0))

    # Bin the ground truth evidence word count
    df['gt_bins'] = pd.cut(
        df['ground_truth_evidence_word_count'], bins=bin_edges, include_lowest=True
    )
    gt_grouped = df.groupby('gt_bins')['detection'].value_counts().unstack(fill_value=0)
    gt_grouped['Recall'] = gt_grouped[0] / (gt_grouped[1] + gt_grouped[0])

    # Add ground truth recalls to the corresponding bin
    for i, interval in enumerate(bin_edges[:-1]):
        recalls_gt[i].append(gt_grouped['Recall'].get(interval, 0))


# Step 3: Calculate mean recall and standard deviation for each bin
mean_recalls_predicted = [np.mean(recalls_predicted[i]) for i in range(len(bin_edges) - 1)]
std_recalls_predicted = [np.std(recalls_predicted[i]) for i in range(len(bin_edges) - 1)]
mean_recalls_gt = [np.mean(recalls_gt[i]) for i in range(len(bin_edges) - 1)]

# Step 4: Plot grouped bar charts
x = np.arange(len(bin_edges) - 1)  # Bin positions
width = 0.25  # Width of each bar

# Create the figure
fig, ax = plt.subplots()
ax.bar(x - width/2, mean_recalls_predicted, width, yerr=std_recalls_predicted, capsize=5, label='Explanations from Models (Mean ± STD)', alpha=0.7, color="#FFB000")
ax.bar(x + width/2, mean_recalls_gt, width, label='Evidence from GT', alpha=0.7, color="#648FFF")
bin_labels = [f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges) - 1)]
plt.xticks(x, bin_labels, rotation=45)
plt.xlabel('Evidence Word Count (quantile bins)', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.legend(fontsize=12, loc="lower right")
plt.tight_layout()
plt.show()