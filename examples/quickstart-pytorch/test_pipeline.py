import torch
from pytorchexample.task import Net, get_client_dataloaders_dirichlet, test, train

def test_pipeline():
    print("Testing data loading and model architecture directly...\n")
    
    alpha = 1.0 # Standard testing alpha
    batch_size = 32
    partition_id = 0
    num_partitions = 10
    
    print(f"Generating Dirichlet partitions (alpha={alpha})...")
    # This will trigger download_and_extract()
    trainloader, valloader = get_client_dataloaders_dirichlet(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        alpha=alpha
    )
    
    print(f"Partition {partition_id}: Train batches={len(trainloader)}, Val batches={len(valloader)}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    print("\nStarting local training loop test (1 epoch)...")
    train_loss = train(model, trainloader, local_epochs=1, lr=0.001, device=device)
    print(f"Train Loss: {train_loss:.4f}")
    
    print("\nStarting validation test...")
    val_loss, val_acc = test(model, valloader, device=device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print("\nPipeline test complete!")

if __name__ == "__main__":
    test_pipeline()
