---
tags: [wearable, activity-monitoring, wisdm]
dataset: [WISDM]
framework: [torch, torchvision, flower]
---

# Privacy-Preserving Wearable Activity Monitoring (Federated Learning)

This project demonstrates privacy-preserving wearable activity monitoring using Federated Learning with PyTorch and Flower. It uses the WISDM smartwatch accelerometer dataset (18 activity classes, 3-channel 128-timestep windows), partitioned across 10 virtual clients in a non-IID Dirichlet setup with advanced score-based client selection.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-pytorch . \
        && rm -rf _tmp \
        && cd quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run the Wearable Demo Experiment

To run the custom experiment orchestrator that executes a 5-round Random Selection baseline followed by a 5-round Score-Based selection (our fairness-aware algorithm), simply use standard Python:

```bash
python run_experiments.py
```

This script will automatically:
1. Orchestrate all 10 clients and Server logic.
2. Accumulate accuracy and loss numbers round-by-round.
3. Save the results to `experiment_results.csv`.
4. Generate the side-by-side comparison plot `fl_comparison.png`.

> [!TIP]
> If you wish to quickly verify the memory footprint and CPU latency of the Activity Monitoring model for smartwatch deployments, run the hardware feasibility script:
> ```bash
> python hardware_check.py
> ```

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 5x`ClientApp` will run in parallel in the available GPU. You can tweak the degree of parallelism by adjusting the settings of this federation in the `pyproject.toml`.

```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

> [!TIP]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
