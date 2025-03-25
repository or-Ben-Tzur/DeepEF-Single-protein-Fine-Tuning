#This script evaluates the model on protein variants
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
sys.path.append('..')    # add parent directory to path
from model.hydro_net import PEM
from model.model_cfg import CFG
import scipy.stats as stats
from tqdm import tqdm
from Utils.train_utils import *


# Constants (similar to those in evaluate.py)
COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'prott5_embeddings'
NANO_TO_ANGSTROM = 0.1
MINI_BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAINED_MODEL_PATH = "./res/trianed_models-2cycle_drop/25_final_model.pt"
REG_LAMBDA = 0.01

# Directories
tensor_root_dir = "./protein_tensors/"
variants_root_dir = "./variants_datasets/"
output_file = "protein_initial_evaluation.csv"

# Function to normalize batch (same as in evaluate.py)
def normalize_batch(batch, LLM_EMB=True):
    batch['one_hot'] = batch['one_hot'][:, :, :, :-1]
    batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM
    if not LLM_EMB:  # zero prot5 embedding
        batch['prott5'] = torch.zeros_like(batch['prott5'])
    return batch

# Dataset class (similar to AllProteinValidationDataset in evaluate.py)
class ProteinEvaluationDataset(Dataset):
    def __init__(self, tensor_root_dir, variants_root_dir, one_mut=True):
        self.tensor_root_dir = tensor_root_dir
        self.variants_root_dir = variants_root_dir
        self.protein_dirs = [protein for protein in os.listdir(self.tensor_root_dir)]
        self.one_mut = one_mut  # remove mutations with more than one mutation

    def __len__(self):
        return len(self.protein_dirs)

    def __getitem__(self, idx):
        protein_dir = os.path.join(self.tensor_root_dir, self.protein_dirs[idx])
        variants_path = os.path.join(self.variants_root_dir, f'{self.protein_dirs[idx]}.csv')
        
        # Check if variants file exists
        if not os.path.exists(variants_path):
            print(f"Warning: variants file not found: {variants_path}")
            # Return empty data
            return {
                'name': self.protein_dirs[idx],
                'mutations': [],
                'prott5': torch.tensor([]),
                'coords': torch.tensor([]),
                'one_hot': torch.tensor([]),
                'delta_g': torch.tensor([]),
                'masks': torch.tensor([])
            }
            
        # Load variants data
        variants = pd.read_csv(variants_path)
        
        # Filter out insertions and deletions
        if 'mut_type' in variants.columns:
            variants = variants[~variants['mut_type'].str.contains('ins|del', na=False)].reset_index(drop=True)
        
        # Load and preprocess the data for each protein
        try:
            coords_tensor = torch.load(os.path.join(protein_dir, COORDS),weights_only=False, map_location=CFG.device)
            delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G), weights_only=False, map_location=CFG.device)
            mask_tensor = torch.load(os.path.join(protein_dir, MASKS), weights_only=False, map_location=CFG.device)
            one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT), weights_only=False, map_location=CFG.device)
            embedding_tensor = self.load_embedding_tensor(os.path.join(protein_dir, PROTT5_EMBEDDINGS))
            
            # Remove mutations with more than one mutation
            if self.one_mut and 'mut_type' in variants.columns:
                one_mut_index = variants[~variants['mut_type'].str.contains(':')]
                variants = variants.loc[one_mut_index.index]
                delta_g_tensor = delta_g_tensor[one_mut_index.index]
                one_hot_tensor = one_hot_tensor[one_mut_index.index]
                embedding_tensor = embedding_tensor[one_mut_index.index]
                
            mutations_data = {
                'name': self.protein_dirs[idx],
                'mutations': variants['mut_type'].to_list() if 'mut_type' in variants.columns else [],
                'prott5': embedding_tensor,
                'coords': coords_tensor,
                'one_hot': one_hot_tensor,
                'delta_g': delta_g_tensor,
                'masks': mask_tensor
            }
            
            return mutations_data
        except Exception as e:
            print(f"Error loading data for protein {self.protein_dirs[idx]}: {e}")
            # Return empty data on error
            return {
                'name': self.protein_dirs[idx],
                'mutations': [],
                'prott5': torch.tensor([]),
                'coords': torch.tensor([]),
                'one_hot': torch.tensor([]),
                'delta_g': torch.tensor([]),
                'masks': torch.tensor([])
            }

    def load_embedding_tensor(self, embeddings_dir):
        embeddings = []
        if not os.path.exists(embeddings_dir):
            print(f"Warning: Embeddings directory not found: {embeddings_dir}")
            return torch.tensor([])
            
        all_embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, 'prott5_embedding_*.pt')),
                                     key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        for filename in all_embedding_files:
            if filename.endswith('.pt'):
                embedding_tensor = torch.load(filename, weights_only=False, map_location=CFG.device)
                embeddings.append(embedding_tensor)
        
        if not embeddings:
            return torch.tensor([])
        return torch.vstack(embeddings)

# Evaluator class
class ProteinEvaluator:
    def __init__(self, model, dataset, device=DEVICE):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.criterion = nn.L1Loss()
        self.mini_batch_size = MINI_BATCH_SIZE
        
    def evaluate_all_proteins(self):
        """
        Evaluate the model on all proteins and generate evaluation metrics
        """
        self.model.eval()
        results = []
        
        print(f"Evaluating {len(self.dataset)} proteins...")
        
        for idx in tqdm(range(len(self.dataset))):
            protein_data = self.dataset[idx]
            protein_name = protein_data['name']
            
            # Skip if no mutations or data is empty
            if not protein_data['mutations'] or len(protein_data['delta_g']) == 0:
                print(f"Skipping {protein_name}: No mutations or empty data")
                continue
                
            # Prepare batch format similar to dataloader output
            batch = {
                'name': protein_name,
                'mutations': protein_data['mutations'],
                'prott5': protein_data['prott5'].unsqueeze(0),  # Add batch dimension
                'coords': protein_data['coords'].unsqueeze(0),  # Add batch dimension
                'one_hot': protein_data['one_hot'].unsqueeze(0),  # Add batch dimension
                'delta_g': protein_data['delta_g'].unsqueeze(0),  # Add batch dimension
                'masks': protein_data['masks'].unsqueeze(0)  # Add batch dimension
            }
            
            # Normalize the batch
            batch = normalize_batch(batch)
            
            # Track predictions and targets
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                # Process in mini-batches
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    try:
                        # Get model predictions
                        output, u_energy, f_energy = self.get_deltaG(batch, j)
                        delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                        
                        # Collect predictions and targets
                        all_preds.extend(output.cpu().numpy())
                        all_targets.extend(delta_g.cpu().numpy())
                    except Exception as e:
                        print(f"Error during prediction for {protein_name} at batch {j}: {e}")
                        continue
            
            # Skip if no predictions were made
            if not all_preds:
                print(f"No predictions were made for {protein_name}")
                continue
                
            # Calculate metrics
            num_variants = len(all_preds)
            
            # Calculate correlation if there are enough data points
            if num_variants >= 2:
                pearson_corr, pearson_p = stats.pearsonr(all_preds, all_targets)
                spearman_corr, spearman_p = stats.spearmanr(all_preds, all_targets)
            else:
                pearson_corr = pearson_p = spearman_corr = spearman_p = float('nan')
                
            mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
            rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
            
            # Store results
            protein_result = {
                'protein_name': protein_name,
                'num_variants': num_variants,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'mae': mae,
                'rmse': rmse
            }
            
            results.append(protein_result)
            
        # Create and return the results DataFrame
        results_df = pd.DataFrame(results)
        return results_df
        
    def get_deltaG(self, batch, i):
        """
        Calculate deltaG prediction for a mini-batch
        This function needs implementation from your model architecture
        """
        # Note: This is a placeholder implementation
        # You would need to implement this based on your model's architecture
        
        # Move data to device
        one_hot_minibatch = batch['one_hot'][0, i: i + self.mini_batch_size].to(self.device)
        prott5_embedding_minibatch = batch['prott5'][0, i: i + self.mini_batch_size].to(self.device)
        batch['coords'] = batch['coords'].to(self.device)
        batch['masks'] = batch['masks'].to(self.device)
        
        # Get graph representations
        # Note: These functions need to be implemented based on your codebase
        folded_graph_minibatch = torch.stack(
            [get_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(), 
                       prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])
        
        unfolded_graph_minibatch = torch.stack(
            [get_unfolded_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(),
                              prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])

        all_graph_minibatch = torch.cat([folded_graph_minibatch, unfolded_graph_minibatch], dim=0)

        # Forward pass through the model
        minibatch_energy = self.model(all_graph_minibatch)
        folded_energy = minibatch_energy[:minibatch_energy.size(0) // 2]
        unfolded_energy = minibatch_energy[minibatch_energy.size(0) // 2:]
        
        return unfolded_energy - folded_energy, unfolded_energy, folded_energy

def main():
    # Load the model
    #load model
    # Import the model
    print("Loading model...")
    model = PEM(layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef).to(CFG.device)
    # Upload model weights
    CFG.model_path = '../data/Trained_models/'
    epoch = 25
    model_dict = torch.load(CFG.model_path+f"{epoch}_final_model.pt",map_location=CFG.device,weights_only=False)
    model.load_state_dict(model_dict['model_state_dict'])
    
    # Create dataset
    print("Creating dataset...")
    dataset = ProteinEvaluationDataset(tensor_root_dir, variants_root_dir)
    
    # Create evaluator
    print("Setting up evaluator...")
    evaluator = ProteinEvaluator(model, dataset)
    
    # Evaluate all proteins
    print("Evaluating proteins...")
    results_df = evaluator.evaluate_all_proteins()
    
    # Save results to CSV
    print(f"Saving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    
    print("Evaluation complete!")
    print(f"Total proteins evaluated: {len(results_df)}")
    print(f"Average Pearson correlation: {results_df['pearson_correlation'].mean()}")
    print(f"Average MAE: {results_df['mae'].mean()}")
    
if __name__ == "__main__":
    main()