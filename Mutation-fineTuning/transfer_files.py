# Description: This script copies protein folders and mutation CSV files from the DeepPEF folder to our folder.

import os
import shutil
import pandas as pd
import re

# Define paths
source_training_dir = "/cs/casp15/Shahar/DeepPEF/data/Processed_K50_dG_datasets/training_data/"
source_mutation_dir = "/cs/casp15/Shahar/DeepPEF/data/Processed_K50_dG_datasets/mutation_datasets/"
mega_val_path = "/cs/casp15/Shahar/DeepPEF/data/Processed_K50_dG_datasets/Pnas_filtering/mega_val.csv"

target_protein_dir = "/cs/casp15/orbentz/protein_tensors/"
target_mutation_dir = "/cs/casp15/orbentz/mutation_datasets/"

# Ensure target directories exist
os.makedirs(target_protein_dir, exist_ok=True)
os.makedirs(target_mutation_dir, exist_ok=True)

# Read the mega_val.csv file
df = pd.read_csv(mega_val_path, sep='\t')

# Extract unique protein names
# The format is "PROTEIN_NAME.pdb_MUTATION" (e.g., "HEEH_KT_rd6_1415.pdb_E38A")
protein_names = set()

for name in df['name']:
    # Extract the part before ".pdb"
    match = re.match(r'(.+?)\.pdb', name)
    if match:
        protein_names.add(match.group(1))

# Also check WT_name column if it exists
if 'WT_name' in df.columns:
    for name in df['WT_name']:
        if pd.notna(name):
            match = re.match(r'(.+?)\.pdb', name)
            if match:
                protein_names.add(match.group(1))

print(f"Found {len(protein_names)} unique proteins in mega_val.csv")

# Copy the folders and CSV files
copied_proteins = 0
copied_mutations = 0

for protein in protein_names:
    # Copy protein folder
    source_protein_path = os.path.join(source_training_dir, protein)
    target_protein_path = os.path.join(target_protein_dir, protein)
    
    if os.path.exists(source_protein_path):
        if not os.path.exists(target_protein_path):
            shutil.copytree(source_protein_path, target_protein_path)
            copied_proteins += 1
            print(f"Copied protein folder: {protein}")
        else:
            print(f"Protein folder already exists in target: {protein}")
    else:
        print(f"Warning: Protein folder not found: {protein}")
    
    # Copy mutation CSV file
    source_mutation_path = os.path.join(source_mutation_dir, f"{protein}.csv")
    target_mutation_path = os.path.join(target_mutation_dir, f"{protein}.csv")
    
    if os.path.exists(source_mutation_path):
        if not os.path.exists(target_mutation_path):
            shutil.copy2(source_mutation_path, target_mutation_path)
            copied_mutations += 1
            print(f"Copied mutation file: {protein}.csv")
        else:
            print(f"Mutation file already exists in target: {protein}.csv")
    else:
        print(f"Warning: Mutation file not found: {protein}.csv")

print(f"\nSummary:")
print(f"Total proteins found in mega_val.csv: {len(protein_names)}")
print(f"Protein folders copied: {copied_proteins}")
print(f"Mutation files copied: {copied_mutations}")
print(f"Done!")