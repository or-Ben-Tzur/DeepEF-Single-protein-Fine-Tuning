from dataloaders.single_protein_dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    csv_path="./variants_datasets/1A0N.csv",
    tensor_folder="./protein_tensors/1A0N"
)


# Check if it's loading batches correctly
for batch in train_loader:
    print("Batch keys:", batch.keys())
    print("Coords shape:", batch["coords"].shape)
    print("Embedding shape:", batch["embedding"].shape)
    print("deltaG shape:", batch["deltaG"].shape)
    break  # Only print first batch