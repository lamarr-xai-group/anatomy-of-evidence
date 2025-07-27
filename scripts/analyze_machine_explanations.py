import pickle
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotting import *


def group_ids_to_spans(id_list):
    """ 
    Groups consecutive IDs into spans.
    Returns: List of lists.
    """
    return [list(group) for _, group in groupby(id_list, key=lambda x: x - id_list.index(x))]


def is_in_sequence(seq, id_list):
    """ 
    Checks if any ID in the sequence is present in the provided ID list.
    Args:
        seq (list): Sequence of IDs to check.
        id_list (list): List of IDs to check against.
    Returns: bool - True if any ID in the sequence is found in the ID list, False otherwise.
    """
    start_id = seq[0] 
    end_id = seq[-1] 
    for i in id_list: 
        if start_id <= i <= end_id:
            return True
    return False


def is_in_context(seqs, pred_id, context_window):
    """ 
    Checks if a predicted token ID is in a context window.
    Args:
        seqs (list): List of sequences (each a list of IDs).
        pred_id (int): Predicted ID to check.
        context_window (int): Size of the context window.
    Returns: bool - True if the predicted token ID is in context window, False otherwise.
    """
    for seq in seqs:
        start_id = seq[0] - context_window
        end_id = seq[-1] + context_window
        if start_id <= pred_id <= end_id:
            return True
    return False
        

def is_prox(gt, pred):
    """ 
    Checks if the predicted IDs are proximate to the ground truth sequences.
    Args:
        gt (list): Ground truth list of token IDs.
        pred (list): List of predicted token IDs.
    Returns: bool - True if all predicted token IDs are in the context window of ground truth sequence, False otherwise.
    """
    # strong assumption: no broken sequences in ground truth 
    gt_sequences = group_ids_to_spans(gt)
    # both for loops are required because the first makes sure that all sequences have a pred match
    for span in gt_sequences: 
        # all sequences have at lease one predicted token match
        if not is_in_sequence(span, pred):
            return False
    for pred_id in pred:
        # if there is one id in pred which is not in a context window of the sequences then false
        if not is_in_context(gt_sequences, pred_id, 10):
            return False
    # all ids are in a context window of a sequence
    return True
        

def match_eval(gt, pred):
    """ 
    Evaluates the match type between ground truth and predicted IDs.
    Args:
        gt (list): Ground truth list of IDs.
        pred (list): List of predicted IDs.
    Returns:
        int: Match type indicator:
            0 - Empty
            1 - Exact match
            2 - Proximate match
            3 - Partial match
            4 - No match
    """
    # no prediction: attribution scores are all below threshold
    if len(pred) == 0: 
        return 0
    gt_sorted = sorted(gt)
    pred_sorted = sorted(pred)
    intersecting = any(check in gt_sorted for check in pred_sorted)
    # exact match: gt and pred contain the same token ids
    if gt_sorted == pred_sorted: 
        return 1
    # overlap/intersection
    if intersecting: 
        if is_prox(gt_sorted, pred_sorted):
            return 2
        else:  # at least one gt token is not met, or at least one pred token is out of context
            return 3
    # no match
    else: 
        return 4


def is_predicted(y_prob):
    """ 
    Checks probability for code being predicted (1) or not (0)
    """
    if y_prob >= 0.5:
        return 1
    else:
        return 0


with open('ModelResultsWithEvidenceAllSupervisedGood.pkl', 'rb') as supervised_file: 
    super_df = pickle.load(supervised_file)

with open('ModelResultsWithEvidenceAllUnsupervised.pkl', 'rb') as unsupervised_file: 
    unsuper_df = pickle.load(unsupervised_file)

# ----------------- Evaluate match types -----------------
super_df['eval'] = super_df.apply(lambda row: match_eval(list(row['evidence_token_ids']), list(row['predicted_evidence_token_ids'])), axis=1)
unsuper_df['eval'] = unsuper_df.apply(lambda row: match_eval(list(row['evidence_token_ids']), list(row['predicted_evidence_token_ids'])), axis=1)

# ----------------- Probability scores -----------------
summary = unsuper_df.groupby('eval')['y_prob'].agg(['mean', 'std']).reset_index()
summary.rename(columns={'mean': 'Average y_prob', 'std': 'Standard Deviation'}, inplace=True)

# ----------------- Plot heatmap of supervised and unsupervised match count alignment -----------------
# Merging data frames
merged_df = pd.merge(super_df, unsuper_df, on=['note_id', 'prediction'], suffixes=('_super', '_unsuper'))
merged_df = merged_df.loc[:,['eval_super', 'eval_unsuper']]

# Count occurrences of each pair (eval_super, eval_unsuper)
pair_counts = merged_df.groupby(['eval_super', 'eval_unsuper']).size().reset_index(name='count')
df = pd.DataFrame(pair_counts)

heatmap_data = df.pivot_table(index='eval_super', columns='eval_unsuper', values='count', fill_value=0)
heatmap_data = heatmap_data.astype(int)

plt.figure(figsize=(8, 6))

fig, ax = plt.subplots(figsize=(10, 10))
fig.set_size_inches(w=2.6, h=2.6)

sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='viridis', cbar=False, linewidths=0.5, square=True,  
            xticklabels=['Empty', 'Exact', 'Pros', 'Partial', 'No Match'],
            yticklabels=['Empty', 'Exact', 'Pros', 'Partial', 'No Match'])

plt.xlabel('Unsupervised')
plt.ylabel('Supervised')
plt.xticks(ticks=np.arange(5) + 0.5, labels=['Empty', 'Exact', 'Prox', 'Partial', 'No Match'])
plt.yticks(ticks=np.arange(5) + 0.5, labels=['Empty', 'Exact', 'Prox', 'Partial', 'No Match'], rotation=0)

plt.tight_layout()
plt.show()

# ----------------- Plot match counts -----------------
value_counts_super = super_df['eval'].value_counts().sort_index()
value_counts_unsuper = unsuper_df['eval'].value_counts().sort_index()

all_values = range(5)
fig, ax = plt.subplots(figsize=(10, 6))

fig.set_size_inches(w=3.2, h=2.3)
bar_width = 0.37
index = index = np.arange(len(all_values))
# Bar plots for each model
bars1 = ax.bar(index, value_counts_super, bar_width, label='supervised', color='#FFB000') 
bars2 = ax.bar(index + bar_width, value_counts_unsuper, bar_width, label='unsupervised', color='#648FFF')

# Add annotations to bars
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, int(yval), va='bottom', ha='center', fontsize=7)

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, int(yval), va='bottom', ha='center', fontsize=7)

match_names = ['not predicted', 'exact match', 'prox match', 'partial match', 'no match']
match_names_red = ['empty', 'exact', 'prox.', 'partial', 'no match']
ax.set_ylabel('Counts')
ax.set_ylim(top=320)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(match_names_red)
ax.legend()

plt.tight_layout()
plt.show()

