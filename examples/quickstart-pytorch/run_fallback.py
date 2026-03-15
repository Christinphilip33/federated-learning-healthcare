import os
import torch
import numpy as np
from flwr.app import start_simulation
from flwr.server import ServerConfig
from flwr.server.app import _run_serverapp_on_simulation  # Hack for newer flwr
from pytorchexample.server_app import app as server_app
from pytorchexample.client_app import app as client_app

def run():
    print("Starting Flower simulation programmatically...")
    
    server_config = ServerConfig(num_rounds=2)
    
    # Map the config values from pyproject.toml manually
    run_config = {
        "num-server-rounds": 2,
        "fraction-evaluate": 0.5,
        "local-epochs": 5,
        "learning-rate": 0.001,
        "batch-size": 32,
        "dirichlet-alpha": 10.0,
        "num-partitions": 10,
        "selection-mode": "score",
        "selected-client-ids": 0,
        "clients-per-round": 5,
    }
    
    # We must start the simulation using the ServerApp and ClientApp
    # Since start_simulation takes a client_fn historically, but we have a ClientApp,
    # we use the modern `flwr run` programmatic equivalent or older start_simulation mappings
    # For flwr >= 1.9, simulation can be invoked via `from flwr.simulation import start_simulation`
    
    try:
        from flwr.simulation import start_simulation
    except ImportError:
        pass

    print("Attempting to run simulation with 10 supernodes...")
    
    # In newer flwr, you often can't just pass `ClientApp` to `start_simulation` easily without the Simulation Engine
    # So we will try to use the CLI programmatically but force the path
    
    os.environ["PATH"] += os.pathsep + r"C:\Users\Christy's Thinkpad\AppData\Roaming\Python\Python310\Scripts"
    
    print("Executing `flwr run .` via `subprocess` with a patched PATH...")
    import subprocess
    import sys
    
    flwr_exe = r"C:\Users\Christy's Thinkpad\AppData\Roaming\Python\Python310\Scripts\flwr.exe"
    
    process = subprocess.Popen(
        [flwr_exe, "run", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    
    process.wait()

if __name__ == "__main__":
    run()
