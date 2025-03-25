import os
import torch
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') 


class SingleProteinDataset(Dataset):
    def __init__(self, csv_path, tensor_folder, indices=None):
        """
        Dataset for one protein and its variants.
        - csv_path: path to the CSV file with variant info (e.g. deltaG, aa_seq, etc.)
        - tensor_folder: folder with tensors (coords, embeddings, etc.)
        - indices: optional list of row indices to include (for splitting)
        """
        self.csv = pd.read_csv(csv_path)
        self.tensor_folder = tensor_folder

        self.coords = torch.load(os.path.join(tensor_folder, "coords_tensor.pt"), map_location=device, weights_only=False)
        self.one_hot = torch.load(os.path.join(tensor_folder, "one_hot_encodings.pt"), map_location=device, weights_only=False)
        self.deltaG = torch.load(os.path.join(tensor_folder, "deltaG.pt"), map_location=device, weights_only=False)
        self.mask = torch.load(os.path.join(tensor_folder, "mask_tensor.pt"), map_location=device, weights_only=False)
        self.embedding_tensor = self.load_embedding_tensor(os.path.join(tensor_folder, 'prott5_embeddings'))

        self.indices = indices if indices is not None else list(range(len(self.csv)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            "coords": self.coords[real_idx],
            "one_hot": self.one_hot[real_idx],
            "deltaG": self.deltaG[real_idx],
            "mask": self.mask[real_idx],
            "embedding": self.embeddings[real_idx],
            "csv_data": self.csv.iloc[real_idx].to_dict()
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
                embedding_tensor = torch.load(filename, weights_only=False, map_location=device)
                embeddings.append(embedding_tensor)
        
        if not embeddings:
            return torch.tensor([])
        return torch.vstack(embeddings)


def create_dataloaders(csv_path, tensor_folder, batch_size=32, num_workers=0, seed=42):
    """
    Randomly split the dataset and create PyTorch DataLoaders.
    Returns: train_loader, val_loader, test_loader
    """
    df = pd.read_csv(csv_path)
    all_indices = list(range(len(df)))

    train_idx, temp_idx = train_test_split(all_indices, train_size=0.8, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

    train_dataset = SingleProteinDataset(csv_path, tensor_folder, train_idx)
    val_dataset   = SingleProteinDataset(csv_path, tensor_folder, val_idx)
    test_dataset  = SingleProteinDataset(csv_path, tensor_folder, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


