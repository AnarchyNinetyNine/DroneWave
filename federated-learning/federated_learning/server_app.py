# server_app.py
"""Flower ServerApp for YOLOv12 with weighted metric aggregation."""

from typing import List, Tuple, Dict, Optional
import numpy as np
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context

from flower_tutorial.task import load_model, get_weights


def weighted_average(metrics: List[Tuple[float, int]]) -> Optional[float]:
    """
    Compute weighted average of (metric, num_examples) pairs.
    Flower expects a function that takes a list of (value, num_examples).
    Return aggregated float or None if input empty.
    """
    if not metrics:
        return None
    # metrics is a list of tuples (value, num_examples)
    total_examples = sum([num for _, num in metrics])
    if total_examples == 0:
        # Avoid division by zero -> simple average
        return float(np.mean([val for val, _ in metrics if val is not None]))
    weighted_sum = 0.0
    count = 0.0
    for val, n in metrics:
        if val is None:
            continue
        weighted_sum += float(val) * float(n)
        count += float(n)
    if count == 0:
        return None
    return float(weighted_sum / count)


def server_fn(context: Context) -> ServerAppComponents:
    """Create ServerAppComponents with FedAvg and weighted metric aggregation."""
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))

    # load initial model and parameters
    model = load_model()
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        # Aggregate per-client evaluate metrics (list of (value, num_examples))
        evaluate_metrics_aggregation_fn=weighted_average,
        # Aggregate per-client fit metrics (train metrics) — Flower will call this for each metric key
        fit_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp object required by flower-superlink
app = ServerApp(server_fn=server_fn)
