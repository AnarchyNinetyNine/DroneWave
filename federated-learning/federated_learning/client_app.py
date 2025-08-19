# client_app.py
"""Flower ClientApp for YOLOv12 (NumPyClient style)."""

import os
import torch
import numpy as np
from flwr.client import NumPyClient, ClientApp
from flwr.common import Parameters
from flwr.common.typing import NDArrays
from flwr.common import Context

from flower_tutorial.task import (
    load_model,
    resolve_data_yaml,
    count_train_examples_from_yaml,
    get_weights,
    set_weights,
    train,
    evaluate,
)


class YoloClient(NumPyClient):
    def __init__(self, model, data_yaml: str, local_epochs: int):
        self.model = model
        self.data_yaml = data_yaml
        self.local_epochs = local_epochs
        # choose device string acceptable by ultralytics
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # move underlying torch model to device for safety (ultralytics handles internally)
        try:
            self.model.model.to(self.device)
        except Exception:
            pass
        # sample counts
        self.num_examples = count_train_examples_from_yaml(self.data_yaml)

    # NumPyClient API:
    def get_parameters(self) -> NDArrays:
        """Return model weights as list of numpy arrays (for initialization)."""
        return get_weights(self.model)

    def fit(self, parameters: NDArrays, config: dict):
        """Fit model on local data, return updated parameters, sample count, and fit metrics."""
        # 1) load parameters into local model
        set_weights(self.model, parameters)

        # 2) train (verbose True -> Ultralytics prints progress)
        metrics = train(
            model=self.model,
            data_yaml=self.data_yaml,
            epochs=self.local_epochs,
            device=self.device,
            verbose=True,
        )

        # 3) After training, extract new weights
        new_weights = get_weights(self.model)

        # 4) include number of examples to enable weighted aggregation on server
        num_examples = self.num_examples if self.num_examples > 0 else 0

        # 5) return (ndarrays, num_examples, metrics)
        return new_weights, num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: dict):
        """Evaluate current model on local validation set and return loss, num_examples, metrics."""
        set_weights(self.model, parameters)
        metrics = evaluate(self.model, self.data_yaml, self.device)
        num_examples = count_train_examples_from_yaml(self.data_yaml) or 0
        # Flower expects (loss, num_examples, metrics) for NumPyClient.evaluate
        # We do not have a scalar "loss" for detection that maps well, so return 0.0
        return 0.0, num_examples, metrics


def client_fn(context: Context):
    """Factory that yields a configured YoloClient based on node_config"""
    # Read partition and run config from the supernode's node_config / run_config
    partition_id = int(context.node_config.get("partition-id", 0))
    num_partitions = int(context.node_config.get("num-partitions", 1))
    local_epochs = int(context.run_config.get("local-epochs", 1))

    data_yaml = resolve_data_yaml(partition_id, num_partitions)
    model = load_model()

    return YoloClient(model=model, data_yaml=data_yaml, local_epochs=local_epochs).to_client()


# Create ClientApp object required by flower-supernode (--clientappio will call this)
app = ClientApp(client_fn)
