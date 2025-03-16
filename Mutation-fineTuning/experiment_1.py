import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model.hydro_net import PEM  # Import your model
from model.model_cfg import CFG  # Import model config
from Utils.train_utils import load_checkpoint  # Load saved model weights

from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(file_path, test_size=0.3, val_size=0.5, random_state=42):
    """
    Splits the dataset into train, validation, and test sets.
    
    Args:
    - file_path (str): Path to the CSV file.
    - test_size (float): Proportion of data for testing (default 30%).
    - val_size (float): Proportion of the remaining test data for validation (default 50% of test set).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - train_df (DataFrame): Training data.
    - val_df (DataFrame): Validation data.
    - test_df (DataFrame): Test data.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Split into train (70%) and test (30%)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Further split test data into validation (15%) and test (15%)
    val_df, test_df = train_test_split(test_df, test_size=val_size, random_state=random_state)

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    return train_df, val_df, test_df


# Define Dataset Class
class ProteinMutationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.tensor([row["deltaG"], row["ddG_ML"]], dtype=torch.float32)  
        label = torch.tensor(1 if row["Stabilizing_mut"] else 0, dtype=torch.float32)
        return features, label


# Load dataset
file_path = r"Mutation-fineTuning\mutation_data\1A0N.csv"
df = pd.read_csv(file_path)

# Split dataset (Assuming test set is 30% of data)
_, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Create DataLoader
test_loader = DataLoader(ProteinMutationDataset(test_df), batch_size=32, shuffle=False)

# Load Pretrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate).to(device)

# Load model weights
CFG.model_path = './data/Trained_models/'
epoch = 25
model_dict = torch.load(CFG.model_path+f"{epoch}_final_model.pt",map_location=CFG.device,weights_only=False)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

# Evaluate Model
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        predicted = torch.round(torch.sigmoid(outputs))  # Convert to binary 0 or 1
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
