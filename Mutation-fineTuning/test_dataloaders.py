from dataloaders.single_protein_dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    csv_path="/cs/casp15/meytav/mutation_datasets/1I6C.csv",
    tensor_folder="/cs/casp15/meytav/protein_tensors/1I6C/"
)
