"""pytorchexample: A Flower / PyTorch app."""

import torch
import numpy as np

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict, MessageType
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test


class CustomFedAvg(FedAvg):
    """
    Custom FedAvg that:
    - Selects clients using *logical ids* (0..num_clients-1)
    - Maps logical ids -> Flower internal node ids
    - Counts how many times each logical client was selected
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_count: dict[int, int] = {}  # logical_id -> count
        self.client_scores: dict[int, float] = {}  # logical_id -> score
        self.node_to_logical: dict[int, int] = {}  # node_id -> logical_id

    def _get_logical_mapping(self, grid: Grid) -> dict[int, int]:
        """
        Create a stable mapping:
          logical_id 0..N-1  ->  actual Flower node_id (big integers)
        We do this by sorting node_ids so it stays stable for one run.
        """
        node_ids = sorted(list(grid.get_node_ids()))
        return {i: node_ids[i] for i in range(len(node_ids))}

    def _parse_selected_ids(self, raw_selected) -> list[int]:
        """
        ConfigRecord values can't be lists, so we pass selected-client-ids as:
        - int: 0
        - str: "0,3,7"
        """
        if isinstance(raw_selected, int):
            return [raw_selected]
        if isinstance(raw_selected, str):
            return [int(x.strip()) for x in raw_selected.split(",") if x.strip() != ""]
        return [0]

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid):
        # Build logical->actual mapping
        logical_to_node = self._get_logical_mapping(grid)
        self.node_to_logical = {v: k for k, v in logical_to_node.items()}
        num_clients = len(logical_to_node)

        # Make sure counters exist
        for lid in range(num_clients):
            if lid not in self.selection_count:
                self.selection_count[lid] = 0

        # Read config
        selection_mode = config.get("selection-mode", "fixed")
        self.current_mode = selection_mode
        raw_selected = config.get("selected-client-ids", 0)
        selected_logical_ids = self._parse_selected_ids(raw_selected)

        clients_per_round = int(config.get("clients-per-round", 5))

        # Decide which logical clients to pick
        all_logical_ids = list(range(num_clients))

        if selection_mode == "fixed":
            chosen_logical = [lid for lid in selected_logical_ids if lid in all_logical_ids]

        elif selection_mode == "even":
            chosen_logical = [lid for lid in all_logical_ids if lid % 2 == 0]

        elif selection_mode == "random":
            rng = np.random.default_rng(server_round)
            k = min(clients_per_round, len(all_logical_ids))
            chosen_logical = rng.choice(all_logical_ids, size=k, replace=False).tolist()

        elif selection_mode == "score":
            print(f"\n[Round {server_round}] Current score table (Logical ID -> Score):")
            print(self.client_scores)

            if not self.client_scores:
                # Round 1: No previous scores exist, fallback to random
                print(f"[Round {server_round}] No scores exist yet. Falling back to random selection.")
                rng = np.random.default_rng(server_round)
                k = min(clients_per_round, len(all_logical_ids))
                chosen_logical = rng.choice(all_logical_ids, size=k, replace=False).tolist()
            else:
                # Rank logical clients based on recorded scores (descending)
                ranked_clients = sorted(
                    self.client_scores.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
                k = min(clients_per_round, len(ranked_clients))
                chosen_logical = [lid for lid, score in ranked_clients[:k]]
        else:
            chosen_logical = all_logical_ids

        # Safety fallback
        if len(chosen_logical) == 0:
            chosen_logical = [0]

        # Convert logical -> actual node ids
        chosen_node_ids = [logical_to_node[lid] for lid in chosen_logical]

        # Update counters
        for lid in chosen_logical:
            self.selection_count[lid] += 1
            
        self.last_selected = chosen_logical

        # Print summary (this is what you show supervisor)
        print(f"\n[Round {server_round}] Selection mode: {selection_mode}")
        print(f"[Round {server_round}] Selected LOGICAL clients: {chosen_logical}")
        print(f"[Round {server_round}] Selection counter so far: {self.selection_count}")

        # Create TRAIN messages
        messages = []
        for node_id in chosen_node_ids:
            content = RecordDict({"arrays": arrays, "config": config})
            msg = grid.create_message(
                content=content,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(msg)

        return messages

    def aggregate_train(self, server_round: int, replies):
        # Let FedAvg do the normal model aggregation first
        aggregated_arrays, aggregated_metrics = super().aggregate_train(server_round, replies)

        # Now compute weighted metrics from client replies
        total_examples = 0
        weighted_acc_sum = 0.0
        weighted_loss_sum = 0.0

        from flwr.common import FitRes

        for reply in replies:
            if not isinstance(reply, FitRes) and hasattr(reply, "content"):
                m = reply.content["metrics"]

                # Extract reply metadata to figure out src node ID
                src_node_id = reply.metadata.src_node_id
                logical_id = self.node_to_logical.get(src_node_id)

                # MetricRecord behaves like a dict in most setups, but we stay safe:
                md = m.to_dict() if hasattr(m, "to_dict") else dict(m)

                num_ex = int(md.get("num-examples", 0))
                acc = float(md.get("local_train_acc", 0.0))
                loss = float(md.get("train_loss", 0.0))
                client_score = float(md.get("client_score", 0.0))

                # Track score for future selection modes
                if logical_id is not None:
                    self.client_scores[logical_id] = client_score

                total_examples += num_ex
                weighted_acc_sum += num_ex * acc
                weighted_loss_sum += num_ex * loss

        if total_examples > 0:
            weighted_train_acc = weighted_acc_sum / total_examples
            weighted_train_loss = weighted_loss_sum / total_examples
        else:
            weighted_train_acc = 0.0
            weighted_train_loss = 0.0

        # ---- Selected prediction score (simple formula) ----
        # Example formula:
        # score = accuracy / (loss + small number)
        eps = 1e-8
        selected_prediction_score = weighted_train_acc / (weighted_train_loss + eps)

        print(f"\n[Round {server_round}] Weighted train acc: {weighted_train_acc:.4f}")
        print(f"[Round {server_round}] Weighted train loss: {weighted_train_loss:.4f}")
        print(f"[Round {server_round}] Selected prediction score: {selected_prediction_score:.4f}\n")
        
        self.last_avg_score = selected_prediction_score

        # Add these new aggregated metrics so they show up in logs/results
        extra = MetricRecord({
            "weighted_train_acc": float(weighted_train_acc),
            "weighted_train_loss": float(weighted_train_loss),
            "selected_prediction_score": float(selected_prediction_score),
        })

        # Merge with whatever FedAvg already returned
        merged = MetricRecord({**dict(aggregated_metrics), **dict(extra)})

        return aggregated_arrays, merged

    def evaluate(self, server_round: int, parameters):
        loss_metrics = super().evaluate(server_round, parameters)
        if loss_metrics is not None and server_round > 0:
            loss, metrics = loss_metrics
            acc = metrics.get("accuracy", 0.0)
            mode = getattr(self, "current_mode", "unknown")
            selected = getattr(self, "last_selected", [])
            avg_score = getattr(self, "last_avg_score", 0.0)

            import csv, os
            file_exists = os.path.isfile("experiment_results.csv")
            with open("experiment_results.csv", "a", newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["selection_mode", "round", "global_acc", "global_loss", "selected_clients", "avg_score"])
                writer.writerow([mode, server_round, acc, loss, str(selected), avg_score])
        return loss_metrics

    def summary(self) -> None:
        print("\n========== FINAL CLIENT SELECTION COUNTS ==========")
        for lid in sorted(self.selection_count.keys()):
            print(f"Client {lid}: selected {self.selection_count[lid]} times")
        print("==================================================\n")


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = CustomFedAvg(fraction_evaluate=fraction_evaluate)

    # IMPORTANT:
    # ConfigRecord values cannot be lists, so selected-client-ids should be int or string like "0,3"
    train_cfg = ConfigRecord({
        "lr": lr,
        "selection-mode": context.run_config.get("selection-mode", "fixed"),
        "selected-client-ids": context.run_config.get("selected-client-ids", 0),
        "clients-per-round": context.run_config.get("clients-per-round", 5),
    })

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_cfg,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset(batch_size=32)
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
