import os
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common import FitRes, Parameters, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from pytorchexample.server_app import CustomFedAvg, global_evaluate
from pytorchexample.task import Net, get_client_dataloaders_dirichlet, train, test

class MockGrid:
    def get_node_ids(self): return list(range(10))
    def create_message(self, content, message_type, dst_node_id, group_id):
        return dst_node_id

class MockClientProxy:
    def __init__(self, cid): self.cid = str(cid)

class MockReply:
    # Acts as a "Message" object for our custom CustomFedAvg logic snippet
    def __init__(self, node_id, content):
        self.metadata = type("Meta", (), {"src_node_id": node_id})()
        self.content = content

def wrapper_evaluate_fn(server_round, parameters, config):
    if not isinstance(parameters, ArrayRecord):
        if isinstance(parameters, Parameters):
            pass
    record = global_evaluate(server_round, parameters)
    return float(record["loss"]), {"accuracy": float(record["accuracy"])}

def run_experiment(selection_mode, num_rounds=5, clients_per_round=5, alpha=0.1):
    print(f"\n=========================================")
    print(f"--- Running Experiment: {selection_mode.upper()} selection ---")
    print(f"=========================================\n")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = Net().to(device)
    arrays = ArrayRecord(global_model.state_dict())
    
    strategy = CustomFedAvg(fraction_evaluate=0.5)
    grid = MockGrid()
    
    config = ConfigRecord({
        "selection-mode": selection_mode,
        "selected-client-ids": 0,
        "clients-per-round": clients_per_round,
        "lr": 0.001
    })
    
    for server_round in range(1, num_rounds + 1):
        messages = strategy.configure_train(server_round, arrays, config, grid)
        selected_nodes = messages 
        
        replies = []
        for node_id in selected_nodes:
            trainloader, _ = get_client_dataloaders_dirichlet(
                partition_id=node_id, num_partitions=10, batch_size=32, alpha=alpha, seed=42
            )
            
            client_model = Net().to(device)
            client_model.load_state_dict(arrays.to_torch_state_dict())
            
            train_loss = train(client_model, trainloader, local_epochs=5, lr=0.001, device=device)
            eval_loss, train_acc = test(client_model, trainloader, device=device)
            
            client_score = train_acc / (train_loss + 1e-8)
            num_ex = int(len(trainloader.dataset))
            content = RecordDict({
                "arrays": ArrayRecord(client_model.state_dict()),
                "metrics": MetricRecord({
                    "train_loss": float(train_loss),
                    "local_train_acc": float(train_acc),
                    "client_score": float(client_score),
                    "num-examples": num_ex
                })
            })
            
            class MockMessage:
                def __init__(self, node_id, content):
                    self.metadata = type("Meta", (), {"src_node_id": node_id})()
                    self.content = content
                def has_error(self): return False
            
            replies.append(MockMessage(node_id, content))
            
        # Call aggregate
        aggregated_arrays, metrics = strategy.aggregate_train(server_round, replies)
        
        # Super() returns parameter tuple (aggregated_parameters, dict)
        if aggregated_arrays:
            arrays = aggregated_arrays
        
        # 4. Evaluate & Log explicitly bypassing super().evaluate
        record = global_evaluate(server_round, arrays)
        loss = record["loss"]
        acc = record["accuracy"]
        avg_score = getattr(strategy, "last_avg_score", 0.0)
        selected = getattr(strategy, "last_selected", [])
        
        file_exists = os.path.isfile("experiment_results.csv")
        with open("experiment_results.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["selection_mode", "round", "global_acc", "global_loss", "selected_clients", "avg_score"])
            writer.writerow([selection_mode, server_round, acc, loss, str(selected), avg_score])
            


def plot_results():
    if not os.path.exists("experiment_results.csv"):
        print("No results file found.")
        return

    df = pd.read_csv("experiment_results.csv")
    
    plt.figure(figsize=(10, 6))
    
    # Plot random
    df_random = df[df['selection_mode'] == 'random']
    if not df_random.empty:
        plt.plot(df_random['round'], df_random['global_acc'], label='Random Selection', marker='o')
        
    # Plot score
    df_score = df[df['selection_mode'] == 'score']
    if not df_score.empty:
        plt.plot(df_score['round'], df_score['global_acc'], label='Score-based Selection', marker='s')
        
    plt.xlabel("Training Rounds")
    plt.ylabel("Global Accuracy")
    plt.title("Wearable Activity Monitoring - FL Selection Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("fl_comparison.png")
    print("\nSaved plot to fl_comparison.png")

if __name__ == "__main__":
    if os.path.exists("experiment_results.csv"):
        os.remove("experiment_results.csv")
        
    run_experiment(selection_mode="random", num_rounds=5, clients_per_round=5, alpha=0.1)
    run_experiment(selection_mode="score", num_rounds=5, clients_per_round=5, alpha=0.1)
    
    plot_results()
