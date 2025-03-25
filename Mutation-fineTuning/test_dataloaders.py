from dataloaders.single_protein_dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    csv_path="./variants_datasets/1A0N.csv",
    tensor_folder="./protein_tensors/1A0N"
)
