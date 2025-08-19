# task.py
"""YOLOv12 task utilities for Flower federation."""

from collections import OrderedDict
from pathlib import Path
import os
import glob
import yaml
import numpy as np
import torch

# Default path to base YOLO weights
YOLO_WEIGHTS = Path("./yolo12n.pt")


def load_model(weights: Path | str = YOLO_WEIGHTS, device: str = "cpu") -> YOLO:
    """
    Load YOLOv12 model with given weights.
    
    Args:
        weights (Path | str): Path to YOLO weights file.
        device (str): Device to load the model on ('cpu', 'cuda', 'mps', etc.).

    Returns:
        YOLO: Loaded YOLO model.
    
    Raises:
        FileNotFoundError: If weights file is not found.
    """
    weights_path = Path(weights).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}\n"
            "Make sure you have downloaded the file and specified the correct path."
        )

    model = YOLO(str(weights_path))
    model.to(device)
    return model


def resolve_data_yaml(partition_id: int, num_partitions: int) -> str:
    """
    Resolve path to data.yaml for this node.
    Preferred layout:
        Data/partition_<id>/data.yaml
    Fallback: Data/data.yaml
    """
    p1 = Path("Data") / f"partition_{partition_id}" / "data.yaml"
    p2 = Path("Data") / "data.yaml"
    if p1.exists():
        return str(p1)
    if p2.exists():
        return str(p2)
    raise FileNotFoundError(f"Couldn't find data.yaml at {p1} or {p2} on this node.")



def count_train_examples_from_yaml(data_yaml_path: str) -> int:
    """Count number of training images from the data.yaml file."""
    with open(data_yaml_path, "r") as f:
        info = yaml.safe_load(f)
    train_spec = info.get("train")
    if train_spec is None:
        return 0
    # `train` can be a single string (folder) or list of files
    if isinstance(train_spec, str):
        # If it's a directory, find images recursively
        train_path = Path(train_spec)
        if not train_path.exists():
            # maybe paths are relative to data_yaml dir
            base = Path(data_yaml_path).parent
            train_path = base / train_spec
        # collect typical image extensions
        files = []
        for ext in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"):
            files.extend(list(train_path.glob(ext)))
        return len(files)
    elif isinstance(train_spec, list):
        # list of image files
        return sum(1 for _ in train_spec)
    else:
        return 0


def parse_val_images_count(data_yaml_path: str) -> int:
    """Count number of validation images."""
    with open(data_yaml_path, "r") as f:
        info = yaml.safe_load(f)
    val_spec = info.get("val")
    if val_spec is None:
        return 0
    if isinstance(val_spec, str):
        base = Path(data_yaml_path).parent
        val_path = Path(val_spec) if Path(val_spec).exists() else base / val_spec
        files = []
        for ext in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"):
            files.extend(list(val_path.glob(ext)))
        return len(files)
    elif isinstance(val_spec, list):
        return sum(1 for _ in val_spec)
    return 0


def get_weights(model: YOLO):
    """Return model parameters as list of NumPy ndarrays (ordered)."""
    state_dict = model.model.state_dict()
    return [val.cpu().numpy() for _, val in state_dict.items()]


def set_weights(model: YOLO, parameters):
    """Load parameters (list of ndarrays) into the model."""
    state_dict = model.model.state_dict()
    if len(list(state_dict.keys())) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(state_dict)} tensors, "
            f"but received {len(parameters)} parameters."
        )
    params_dict = zip(state_dict.keys(), parameters)
    new_state = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.model.load_state_dict(new_state, strict=True)


def train(model: YOLO, data_yaml: str, epochs: int, device: str, verbose: bool = True):
    """
    Train locally and return a dictionary of useful training metrics.

    The Ultralyics `train()` returns a Results object; we will extract some
    values and also run a validation pass at the end for stable metrics.
    """
    # Ensure device string is accepted by ultralytics (e.g., 'cpu' or '0' or 'cuda')
    # ultralytics uses device param similar to PyTorch, accept "cpu" or "cuda" index
    # We'll pass device directly.
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=device,
        verbose=verbose,
        save=True,
    )

    # Attempt to extract training losses if available
    train_metrics = {}
    try:
        rd = results.results_dict  # dictionary of aggregated training metrics
        # rd keys may include: 'train/box_loss','train/cls_loss','train/obj_loss', etc.
        for k, v in rd.items():
            train_metrics[k] = float(v)
    except Exception:
        # fallback: no results_dict
        train_metrics["train/box_loss"] = None
        train_metrics["train/cls_loss"] = None
        train_metrics["train/obj_loss"] = None

    # Run validation to get mAP/precision/recall
    try:
        val_metrics = model.val(data=data_yaml, imgsz=640, device=device, verbose=False)
        # Ultralyics returns an object: val_metrics.box.map50, val_metrics.box.map
        train_metrics["val/mAP50"] = float(val_metrics.box.map50)
        train_metrics["val/mAP"] = float(val_metrics.box.map)  # mAP50-95
        # precision/recall (if present)
        if hasattr(val_metrics, "metrics"):
            # older APIs might contain metrics; attempt safe extraction
            for attr in ("precision", "recall"):
                if hasattr(val_metrics.metrics, attr):
                    train_metrics[f"val/{attr}"] = float(getattr(val_metrics.metrics, attr))
    except Exception:
        # If validation fails, ignore
        train_metrics.setdefault("val/mAP50", None)
        train_metrics.setdefault("val/mAP", None)

    return train_metrics



def evaluate(model: YOLO, data_yaml: str, device: str):
    """
    Evaluate model returning a metrics dict for server aggregation.
    We return: loss (None), mAP50, mAP, precision, recall
    """
    val_metrics = model.val(data=data_yaml, imgsz=640, device=device, verbose=False)
    metrics = {}
    metrics["val/mAP50"] = float(val_metrics.box.map50)
    metrics["val/mAP"] = float(val_metrics.box.map)
    # Precision/recall if available (best-effort)
    try:
        metrics["val/precision"] = float(val_metrics.box.pd) if hasattr(val_metrics.box, "pd") else None
    except Exception:
        metrics["val/precision"] = None
    try:
        # ultralytics may store recall differently; attempt to fetch
        metrics["val/recall"] = float(val_metrics.box.rd) if hasattr(val_metrics.box, "rd") else None
    except Exception:
        metrics["val/recall"] = None

    return metrics
