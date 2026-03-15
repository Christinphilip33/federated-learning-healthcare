import torch
from pytorchexample.task import Net, get_client_dataloaders_dirichlet, test, train

def run_all_clients():
    print("Simulating all 10 clients to compute individual Accuracy and Loss...\n")
    
    alpha = 0.1 # Current dirichlet alpha
    batch_size = 32
    num_partitions = 10
    local_epochs = 5
    lr = 0.001
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize a global model (so all clients start from the same weights, like in FL)
    global_model = Net().to(device)
    initial_weights = global_model.state_dict()
    
    for partition_id in range(num_partitions):
        print(f"--- Client {partition_id} ---")
        
        # Load local data partition
        trainloader, valloader = get_client_dataloaders_dirichlet(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            alpha=alpha
        )
        
        # Create a fresh client model loaded with the global weights
        client_model = Net().to(device)
        client_model.load_state_dict(initial_weights)
        
        # Train locally
        train_loss = train(client_model, trainloader, local_epochs=local_epochs, lr=lr, device=device)
        
        # Evaluate locally
        val_loss, val_acc = test(client_model, valloader, device=device)
        
        # In our server_app, prediction score was: acc / (loss + 1e-8)
        client_score = val_acc / (train_loss + 1e-8)
        
        print(f"Data Samples: {len(trainloader.dataset)} Train, {len(valloader.dataset)} Val")
        print(f"Local Train Loss: {train_loss:.4f}")
        print(f"Local Val Loss:   {val_loss:.4f}")
        print(f"Local Val Acc:    {val_acc:.4f}")
        print(f"Client Score:     {client_score:.4f}\n")

if __name__ == "__main__":
    run_all_clients()
