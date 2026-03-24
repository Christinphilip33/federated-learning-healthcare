"""pytorchexample: Flower / PyTorch ClientApp using WISDM Wearable Data and Dirichlet partitions."""
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from pytorchexample.task import (
    Net,
    train as train_fn,
    test as test_fn,
    get_client_dataloaders_dirichlet,
)

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config.get("num-partitions", context.node_config.get("num-partitions"))
    if num_partitions is None:
        raise KeyError("num-partitions not found in run_config or node_config")

    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]
    alpha = float(context.run_config["dirichlet-alpha"])

    trainloader, _ = get_client_dataloaders_dirichlet(
        partition_id, num_partitions, batch_size, alpha, seed=42
    )

    # 1) Local training
    train_loss = train_fn(model, trainloader, local_epochs, lr, device)

    # 2) Quick local check (optional but useful)
    local_eval_loss, local_train_acc = test_fn(model, trainloader, device)

    # Calculate selection score
    client_score = local_train_acc / (train_loss + 1e-8)

    # 3) Send back weights + metrics
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_loss": float(train_loss),
            "local_eval_loss": float(local_eval_loss),
            "local_train_acc": float(local_train_acc),
            "client_score": float(client_score),

            # IMPORTANT: Flower expects this exact key name
            "num-examples": int(len(trainloader.dataset)),

            "partition_id": int(partition_id),
            "alpha": float(alpha),
        }),
    })
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config.get("num-partitions", context.node_config.get("num-partitions"))
    if num_partitions is None:
        raise KeyError("num-partitions not found in run_config or node_config")

    batch_size = context.run_config["batch-size"]
    alpha = float(context.run_config["dirichlet-alpha"])

    _, valloader = get_client_dataloaders_dirichlet(
        partition_id, num_partitions, batch_size, alpha, seed=42
    )

    eval_loss, eval_acc = test_fn(model, valloader, device)

    # IMPORTANT: do NOT reference train_loss here
    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": float(eval_loss),
            "eval_acc": float(eval_acc),

            # If you want server to weight eval too:
            "num-examples": int(len(valloader.dataset)),

            "partition_id": int(partition_id),
            "alpha": float(alpha),
        }),
    })
    return Message(content=content, reply_to=msg)
