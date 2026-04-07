import random
from Bio import SeqIO

# 1. Data Standardization

datasets_fasta = ["dataset2_Antimicrobial.fasta",
                  "dataset3_Antibacterial.fasta",
                  "uniprot.fasta"]
datasets_txt = "dataset1_APD6.txt"

all_sequences = []

txt_sequences = []
with open(datasets_txt, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            continue
        txt_sequences.append(line)

print(f"{datasets_txt}: {len(txt_sequences)} sequences loaded.")

all_sequences.extend(txt_sequences)

for dataset in datasets_fasta:
    fasta_sequences = [str(record.seq) for record in SeqIO.parse(dataset, "fasta")]
    print(f"{dataset}: {len(fasta_sequences)} sequences loaded.")
    all_sequences.extend(fasta_sequences)

print(f"Total sequences loaded after combining all files: {len(all_sequences)}")

# 2. Filtering Criteria
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

filtered_sequences = []
removed_invalid_length = 0
removed_invalid_chars = 0

for seq in all_sequences:
    if not (10 <= len(seq) <= 50):
        removed_invalid_length += 1
        continue
    if any(aa not in valid_aa for aa in seq):
        removed_invalid_chars += 1
        continue
    filtered_sequences.append(seq)

print(f"\nFiltering step:")
print(f"Removed due to invalid length: {removed_invalid_length}")
print(f"Removed due to invalid characters: {removed_invalid_chars}")
print(f"Remaining sequences after filtering: {len(filtered_sequences)}")


# 3. Deduplication (custom script)
unique_sequences = list(set(filtered_sequences))
removed_duplicates = len(filtered_sequences) - len(unique_sequences)

print(f"\nDeduplication step:")
print(f"Removed duplicates: {removed_duplicates}")
print(f"Final unique sequences: {len(unique_sequences)}")

# 4. Step-by-step removal statistics
print("\nStep-by-step removal statistics:")
print(f"Initial sequences: {len(all_sequences)}")
print(f"Removed invalid length: {removed_invalid_length}")
print(f"Removed invalid characters: {removed_invalid_chars}")
print(f"Removed duplicates: {removed_duplicates}")
print(f"Final sequences: {len(unique_sequences)}")


# 5. Class Balancing Strategy
# Dla przykładu zakładamy, że połowa unikalnych sekwencji jest pozytywna, druga połowa negatywna
# W praktyce użyj własnych etykiet
positive_sequences = unique_sequences[:len(unique_sequences) // 2]
negative_sequences = unique_sequences[len(unique_sequences) // 2:]

# Random Undersampling do 1:1, jeśli klasa negatywna jest większa
if len(negative_sequences) > len(positive_sequences):
    negative_sequences = random.sample(negative_sequences, len(positive_sequences))

print("\nClass balancing:")
print(f"Positive sequences: {len(positive_sequences)}")
print(f"Negative sequences (after undersampling): {len(negative_sequences)}")
print(f"Final ratio (Positive:Negative) = {len(positive_sequences)}:{len(negative_sequences)}")



