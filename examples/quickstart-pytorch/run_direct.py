import flwr as fl
from pytorchexample.server_app import app as server_app
from pytorchexample.client_app import app as client_app

def run_simulation():
    print("Starting Flower Simulation directly via Python API...")
    
    fl.simulation.start_simulation(
        client_app=client_app,
        server_app=server_app,
        num_supernodes=10,
        backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.0}},
    )

if __name__ == "__main__":
    run_simulation()
