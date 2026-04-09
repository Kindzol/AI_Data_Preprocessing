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
allowed_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")


# Filtering

def filter_sequences(sequences):
    filtered = []
    removed_length = 0
    removed_chars = 0
    for seq in sequences:
        if not (10 <= len(seq) <= 50):
            removed_length += 1
            continue
        if any(aa not in allowed_amino_acids for aa in seq):
            removed_chars += 1
            continue
        filtered.append(seq)
    return filtered, removed_length, removed_chars

# Load positive sequences
positive_sequences_before_filtering = []

with open(positive_txt, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        positive_sequences_before_filtering.append(line)
print(f"{positive_txt}: {len(positive_sequences_before_filtering)} sequences loaded.")

for dataset in positive_fasta:
    seqs = [str(record.seq) for record in SeqIO.parse(dataset, "fasta")]
    print(f"{dataset}: {len(seqs)} sequences loaded.")
    positive_sequences_before_filtering.extend(seqs)

print(f"Total raw positive sequences: {len(positive_sequences_before_filtering)}")


# Load negative sequences
negative_sequences_before_filtering = []

seqs = [str(record.seq) for record in SeqIO.parse(negative_fasta, "fasta")]
print(f"{negative_fasta}: {len(seqs)} sequences loaded.")
negative_sequences_before_filtering.extend(seqs)
print(f"Total raw negative sequences: {len(negative_sequences_before_filtering)}")


# Filter both separately
pos_filtered, pos_removed_len, pos_removed_chars = filter_sequences(positive_sequences_before_filtering)
neg_filtered, neg_removed_len, neg_removed_chars = filter_sequences(negative_sequences_before_filtering)


# Step-by-step removal statistics

print(f"\nPositive after filtering: {len(pos_filtered)} "
      f"(removed {pos_removed_len} bad length, {pos_removed_chars} bad chars)")
print(f"Negative after filtering: {len(neg_filtered)} "
      f"(removed {neg_removed_len} bad length, {neg_removed_chars} bad chars)")


# Deduplication

# Deduplicate both separately
positive_set = set(pos_filtered)
negative_set = set(neg_filtered)

negative_set = negative_set - positive_set

unique_positive = list(positive_set)
unique_negative = list(negative_set)


print(f"\nAfter deduplication:")
print(f"  Unique positive (AMP):     {len(unique_positive)}")
print(f"  Unique negative (non-AMP): {len(unique_negative)}")


# Class balancing

# Balance-random undersampling to 1:1
# we cheeck how many samples we can take from both classes
positive_count = len(unique_positive)
negative_count = len(unique_negative)

sample_size = min(positive_count, negative_count)

# we randomly select x sequences from each class
balanced_positive = random.sample(unique_positive, sample_size)
balanced_negative = random.sample(unique_negative, sample_size)

print(f"\nAfter balancing (random undersampling):")
print(f"  Positive: {len(balanced_positive)}")
print(f"  Negative: {len(balanced_negative)}")
print(f"  Ratio Pos:Neg = 1:1  (total: {len(balanced_positive) + len(balanced_negative)})")

all_sequences = balanced_positive + balanced_negative

positive_labels = [1] * len(balanced_positive)
negative_labels = [0] * len(balanced_negative)

all_labels = positive_labels + negative_labels

print("\n~~Ready dataset~~")
print("Total sequences:", len(all_sequences))
print("Total labels:", len(all_labels))


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

    # Amino Acid Composition
    aac = prot.amino_acids_percent
    features.update(aac)

    return features

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
print(f"Class distribution:\n{df['Class'].value_counts()}")

print("\nExtracting features...")
features_df = df['Sequence'].apply(lambda x: pd.Series(extract_features(x)))

final_dataset = pd.concat([df, features_df], axis=1)

print(f"\nFinal dataset skeleton:")
print(f"Columns: {list(final_dataset.columns)}")
print("\nFirst 3 rows:")
print(final_dataset.head(3))

final_dataset.to_csv("final_dataset.csv", index=False)
print("\n Saved to final_dataset.csv")


# visualisation
pos_lengths_before_filtering = []
for sequence in positive_sequences_before_filtering:
    pos_lengths_before_filtering.append(len(sequence))

neg_lengths_before_filtering = []
for sequence in negative_sequences_before_filtering:
    neg_lengths_before_filtering.append(len(sequence))


bal_pos_lengths = []
for sequence in balanced_positive:
    bal_pos_lengths.append(len(sequence))

bal_neg_lengths = []
for sequence in balanced_negative:
    bal_neg_lengths.append(len(sequence))


df_pos = final_dataset[final_dataset['Class'] == 1]
df_neg = final_dataset[final_dataset['Class'] == 0]

amino_acids = list('ACDEFGHIKLMNPQRSTVWY')


# plot 1: sequence lengths before vs after cleaning
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sequence length distribution', fontsize=15, fontweight='bold')
bins = range(0, 220, 5)

# before cleaning
axes[0].hist(pos_lengths_before_filtering, bins=bins, alpha=0.6, color='hotpink', label='Positive (AMP)')
axes[0].hist(neg_lengths_before_filtering, bins=bins, alpha=0.6, color='cornflowerblue',    label='Negative (non-AMP)')
axes[0].axvline(10,  color='black', linestyle='--', linewidth=1, label='Filter bounds (10–50)')
axes[0].axvline(50,  color='black', linestyle='--', linewidth=1)
axes[0].set_title('Before cleaning')
axes[0].set_xlabel('Sequence length (aa)')
axes[0].set_ylabel('Count')
axes[0].set_xlim(0, 200)
axes[0].legend()

# after cleaning
axes[1].hist(bal_pos_lengths, bins=range(10, 51, 1), alpha=0.6, color='hotpink', label='Positive (AMP)')
axes[1].hist(bal_neg_lengths, bins=range(10, 51, 1), alpha=0.6, color='cornflowerblue',    label='Negative (non-AMP)')
axes[1].set_title('After cleaning & balancing')
axes[1].set_xlabel('Sequence length (aa)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('plot1_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# plot 2: comparing the average amino acid composition between the Positive and Negative datasets
avg_pos_aac = df_pos[amino_acids].mean()
avg_neg_aac = df_neg[amino_acids].mean()

x = np.arange(len(amino_acids))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('Average amino acid composition: AMP vs non-AMP', fontsize=15, fontweight='bold')

bars1 = ax.bar(x - width/2, avg_pos_aac * 100, width, label='Positive (AMP)',     color='hotpink', alpha=0.85)
bars2 = ax.bar(x + width/2, avg_neg_aac * 100, width, label='Negative (non-AMP)', color='cornflowerblue',    alpha=0.85)

ax.set_xlabel('Amino acid')
ax.set_ylabel('Average composition (%)')
ax.set_xticks(x)
ax.set_xticklabels(amino_acids)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot2_amino_acid_composition.png', dpi=150, bbox_inches='tight')
plt.show()


# plot 3: comparison of pI
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Comparison of isoelectric point (pI) between AMP and non-AMP', fontsize=15, fontweight='bold')

labels = ['Positive (AMP)', 'Negative (non-AMP)']
colors = ['hotpink', 'cornflowerblue']

pi_data = [df_pos['pI'].values, df_neg['pI'].values]
bp = ax.boxplot(pi_data, tick_labels=labels, patch_artist=True, notch=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('pI')
ax.axhline(7, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Neutral pH (7.0)')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot3_pI.png', dpi=150, bbox_inches='tight')
plt.show()