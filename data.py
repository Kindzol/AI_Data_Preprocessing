import random
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

positive_fasta = ["dataset2_Antimicrobial.fasta", "dataset3_Antibacterial.fasta"]
positive_txt = "dataset1_APD6.txt"
negative_fasta = "uniprot.fasta"

# filter sequences
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")


# Filtering

def filter_sequences(sequences):
    filtered = []
    removed_length = 0
    removed_chars = 0
    for seq in sequences:
        if not (10 <= len(seq) <= 50):
            removed_length += 1
            continue
        if any(aa not in valid_aa for aa in seq):
            removed_chars += 1
            continue
        filtered.append(seq)
    return filtered, removed_length, removed_chars

# Load positive sequences
raw_positive = []

with open(positive_txt, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        raw_positive.append(line)
print(f"{positive_txt}: {len(raw_positive)} sequences loaded.")

for dataset in positive_fasta:
    seqs = [str(record.seq) for record in SeqIO.parse(dataset, "fasta")]
    print(f"{dataset}: {len(seqs)} sequences loaded.")
    raw_positive.extend(seqs)

print(f"Total raw positive sequences: {len(raw_positive)}")

# Load negative sequences
raw_negative = [str(record.seq) for record in SeqIO.parse(negative_fasta, "fasta")]
print(f"\n{negative_fasta}: {len(raw_negative)} sequences loaded.")
print(f"Total raw negative sequences: {len(raw_negative)}")


# Filter both separately
filtered_positive, rlen_p, rchar_p = filter_sequences(raw_positive)
filtered_negative, rlen_n, rchar_n = filter_sequences(raw_negative)


# Step-by-step removal statistics

print(f"\nPositive after filtering: {len(filtered_positive)} "
      f"(removed {rlen_p} bad length, {rchar_p} bad chars)")
print(f"Negative after filtering: {len(filtered_negative)} "
      f"(removed {rlen_n} bad length, {rchar_n} bad chars)")


# Deduplication

# Deduplicate both separately
unique_positive = list(set(filtered_positive))
unique_negative = list(set(filtered_negative))

# Also remove any sequences that appear in BOTH (just in case)
positive_set = set(unique_positive)
unique_negative = [seq for seq in unique_negative if seq not in positive_set]


print(f"\nAfter deduplication:")
print(f"  Unique positive (AMP):     {len(unique_positive)}")
print(f"  Unique negative (non-AMP): {len(unique_negative)}")


# Class Balancing

# Balance: Random Undersampling to 1:1
n = min(len(unique_positive), len(unique_negative))
balanced_positive = random.sample(unique_positive, n)
balanced_negative = random.sample(unique_negative, n)

print(f"\nAfter balancing (random undersampling):")
print(f"  Positive: {len(balanced_positive)}")
print(f"  Negative: {len(balanced_negative)}")
print(f"  Ratio Pos:Neg = 1:1  (total: {len(balanced_positive) + len(balanced_negative)})")

# ── Create labels ─────────────────────────────────────────────────────────────
all_sequences = balanced_positive + balanced_negative
all_labels    = [1] * len(balanced_positive) + [0] * len(balanced_negative)

print(f"\nDataset ready for feature extraction and training.")
print(f"   Total sequences: {len(all_sequences)}, Labels: {len(all_labels)}")




# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(seq):
    """Calculates physicochemical properties for a given sequence"""
    prot = ProteinAnalysis(str(seq))

    features = {
        'Molecular_Weight': prot.molecular_weight(),
        'pI': prot.isoelectric_point(),
        'GRAVY': prot.gravy(),
        'Instability_Index': prot.instability_index(),
        'Aromaticity': prot.aromaticity(),
    }

    # Amino Acid Composition (20 features, e.g. A: 0.1, C: 0.05, ...)
    aac = prot.amino_acids_percent
    features.update(aac)

    return features


# ── Build rows ────────────────────────────────────────────────────────────────
rows = []

for i, seq in enumerate(balanced_positive):
    rows.append({
        'Sequence_Name': f'AMP_{i + 1}',
        'Sequence': seq,
        'Class': 1,
        'Sequence_Length': len(seq),
    })

for i, seq in enumerate(balanced_negative):
    rows.append({
        'Sequence_Name': f'nonAMP_{i + 1}',
        'Sequence': seq,
        'Class': 0,
        'Sequence_Length': len(seq),
    })

df = pd.DataFrame(rows)
print(f"Total sequences in dataframe: {len(df)}")
print(f"Class distribution:\n{df['Class'].value_counts()}")

# ── Extract features for every sequence ──────────────────────────────────────
print("\nExtracting features... (may take a moment)")
features_df = df['Sequence'].apply(lambda x: pd.Series(extract_features(x)))

# ── Combine into final dataset ────────────────────────────────────────────────
final_dataset = pd.concat([df, features_df], axis=1)

print(f"\nFinal dataset shape: {final_dataset.shape}")
print(f"Columns: {list(final_dataset.columns)}")
print("\nFirst 3 rows:")
print(final_dataset.head(3))

# ── Save to CSV ───────────────────────────────────────────────────────────────
final_dataset.to_csv("final_amp_dataset.csv", index=False)
print("\n Saved to final_amp_dataset.csv")

# visualisation

# ── Dane potrzebne do wizualizacji ────────────────────────────────────────────
# Długości PRZED filtrowaniem
raw_pos_lengths = [len(s) for s in raw_positive]
raw_neg_lengths = [len(s) for s in raw_negative]

# Długości PO filtrowaniu i balansowaniu
bal_pos_lengths = [len(s) for s in balanced_positive]
bal_neg_lengths = [len(s) for s in balanced_negative]

# Subsetujemy final_dataset na klasy
df_pos = final_dataset[final_dataset['Class'] == 1]
df_neg = final_dataset[final_dataset['Class'] == 0]

amino_acids = list('ACDEFGHIKLMNPQRSTVWY')


# PLOT 1: Sequence Length Distribution (before vs after cleaning)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sequence Length Distribution', fontsize=15, fontweight='bold')

bins = range(0, 220, 5)

# Before cleaning
axes[0].hist(raw_pos_lengths, bins=bins, alpha=0.6, color='hotpink', label='Positive (AMP)')
axes[0].hist(raw_neg_lengths, bins=bins, alpha=0.6, color='cornflowerblue',    label='Negative (non-AMP)')
axes[0].axvline(10,  color='black', linestyle='--', linewidth=1, label='Filter bounds (10–50)')
axes[0].axvline(50,  color='black', linestyle='--', linewidth=1)
axes[0].set_title('Before Cleaning')
axes[0].set_xlabel('Sequence Length (aa)')
axes[0].set_ylabel('Count')
axes[0].set_xlim(0, 200)
axes[0].legend()

# After cleaning & balancing
axes[1].hist(bal_pos_lengths, bins=range(10, 55, 2), alpha=0.6, color='hotpink', label='Positive (AMP)')
axes[1].hist(bal_neg_lengths, bins=range(10, 55, 2), alpha=0.6, color='cornflowerblue',    label='Negative (non-AMP)')
axes[1].set_title('After Cleaning & Balancing')
axes[1].set_xlabel('Sequence Length (aa)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('plot1_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot 1 saved.")


# PLOT 2: Amino Acid Composition comparison (AMP vs non-AMP)
avg_pos_aac = df_pos[amino_acids].mean()
avg_neg_aac = df_neg[amino_acids].mean()

x = np.arange(len(amino_acids))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('Average Amino Acid Composition: AMP vs non-AMP', fontsize=15, fontweight='bold')

bars1 = ax.bar(x - width/2, avg_pos_aac * 100, width, label='Positive (AMP)',     color='hotpink', alpha=0.85)
bars2 = ax.bar(x + width/2, avg_neg_aac * 100, width, label='Negative (non-AMP)', color='cornflowerblue',    alpha=0.85)

ax.set_xlabel('Amino Acid')
ax.set_ylabel('Average Composition (%)')
ax.set_xticks(x)
ax.set_xticklabels(amino_acids)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot2_amino_acid_composition.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot 2 saved.")


# PLOT 3: Physicochemical Comparison — GRAVY & pI boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Physicochemical Properties: AMP vs non-AMP', fontsize=15, fontweight='bold')

labels = ['Positive (AMP)', 'Negative (non-AMP)']
colors = ['hotpink', 'cornflowerblue']

# GRAVY index
gravy_data = [df_pos['GRAVY'].values, df_neg['GRAVY'].values]
bp1 = axes[0].boxplot(gravy_data, tick_labels=labels, patch_artist=True, notch=True)

for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('GRAVY Index (Hydrophobicity)')
axes[0].set_ylabel('GRAVY Score')
axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Hydrophilic | Hydrophobic')
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.3)

# Isoelectric Point (pI)
pi_data = [df_pos['pI'].values, df_neg['pI'].values]
bp2 = axes[1].boxplot(pi_data,    tick_labels=labels, patch_artist=True, notch=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title('Isoelectric Point (pI)')
axes[1].set_ylabel('pI')
axes[1].axhline(7, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Neutral pH (7.0)')
axes[1].legend(fontsize=8)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot3_physicochemical.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot 3 saved.")