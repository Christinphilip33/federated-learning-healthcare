# Federated Learning Project Status

## 1. What We Are Doing
We are building a **Federated Learning (FL) system** using PyTorch and the Flower (`flwr`) framework. The goal is to collaboratively train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset across multiple simulated clients, without raw data leaving the client nodes.

## 2. What We Have Done
- **Project Structure**: Set up a standard Flower structure with `pyproject.toml`, `server_app.py`, `client_app.py`, and `task.py`.
- **Model & Data**: 
  - Created a CNN model (`Net`) in `task.py` for image classification.
  - Implemented **Dirichlet distribution partitioning** (`alpha=10.0`) to simulate non-IID (unbalanced/heterogeneous) data across 10 client partitions.
- **Custom Client Logic**: 
  - `client_app.py` handles local training and evaluation. It computes a custom `client_score` (accuracy/loss) based on local data and sends it back to the server alongside the model weights.
- **Custom Server Aggregator**: 
  - Built a `CustomFedAvg` strategy in `server_app.py`.
  - Added support for multiple client **selection modes** (`fixed`, `even`, `random`, `score`).
  - In `score` mode, the server uses the `client_score` reported by clients to rank and preferentially select the best-performing clients for subsequent rounds.
  - Added custom weighted aggregation for training accuracy and loss.
- **Testing**:
  - Ran local simulations (`flwr run`) to verify the custom training loop and selection logic (as seen in `simulation_output.txt`). The output shows clients reporting their local data distribution and the server computing weighted metrics.

## 3. What Is Exactly Happening Here?
Right now, the simulation is configured to run for **2 rounds**. There are **10 total data partitions** (supernodes), but the server only selects **5 clients per round** (`clients-per-round = 5`). 
During a round:
1. The server requests the selected 5 clients to train globally distributed model weights.
2. The clients use their locally partitioned Dirichlet CIFAR-10 dataset to train, computing their local accuracy and loss.
3. The clients return their updated weights, metrics, and a calculated `client_score`.
4. The server aggregates the weights (using FedAvg), aggregates the metrics to provide global visibility, and records the `client_score` of each client to rank them for selection in the next round.

## 4. Next Steps in the Project
Based on the current foundation, here are logical next steps:
- **Experiment with Non-IID Data:** Lower the `dirichlet-alpha` value (e.g., `0.1` or `1.0`) in `pyproject.toml` to make the data distribution highly skewed, and observe how well the `Global Model` converges using different selection strategies.
- **Evaluate Selection Strategies:** Compare the `score` selection mode against the `random` selection mode. Does intelligently picking clients based on their score improve convergence speed or final accuracy?
- **Increase Scale:** Scale up the simulation to more rounds (e.g., 50 rounds) and more clients (e.g., 100 clients, picking 10 per round) to see how the custom FedAvg strategy performs under scale.
- **Logging and Visualization:** Integrate `TensorBoard` or `Weights & Biases` in `server_app.py` to explicitly plot the `weighted_train_acc` and `selected_prediction_score` over many rounds visually.
