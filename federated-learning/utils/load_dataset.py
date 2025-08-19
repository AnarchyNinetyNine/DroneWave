#!/usr/bin/env python3.10

"""
load_data.py
------------
Downloads a UAV-Gesture dataset version from Roboflow and copies only
the files corresponding to a given partition ID into a structured
Dataset directory for training, validation, and testing.

Usage:
    python load_data.py --api_key <ROBOFLOW_API_KEY> --workspace <WORKSPACE_NAME> \
        --project <PROJECT_NAME> --version <VERSION_NUMBER> --partition <PARTITION_ID>
"""

import os
import shutil
import argparse
from roboflow import Roboflow


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load UAV-Gesture dataset by partition ID.")
    parser.add_argument("--api_key", required=True, help="Roboflow API key")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace name")
    parser.add_argument("--project", required=True, help="Roboflow project name")
    parser.add_argument("--version", required=True, type=int, help="Roboflow project version")
    parser.add_argument("--partition", required=True, type=int, help="Partition ID to filter files")
    return parser.parse_args()


def main():
    args = parse_args()
    partition_prefix = f"S{args.partition}_"

    # -----------------------------
    # Download Dataset
    # -----------------------------
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download("yolov12")

    source_dir = f"./{args.project}-{args.version}"
    dest_dir = f"../Data/partition_{args.partition}"

    # -----------------------------
    # Create Destination Directory
    # -----------------------------
    os.makedirs(dest_dir, exist_ok=True)
    for split in ("train", "valid", "test"):
        for dtype in ("images", "labels"):
            os.makedirs(os.path.join(dest_dir, split, dtype), exist_ok=True)

    # -----------------------------
    # Copy Optional Non-Data Files
    # -----------------------------
    for fname in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
        src_file = os.path.join(source_dir, fname)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_dir)

    # -----------------------------
    # Copy Partition-Specific Files
    # -----------------------------
    for split in ("train", "valid", "test"):
        for dtype in ("images", "labels"):
            src_path = os.path.join(source_dir, split, dtype)
            dest_path = os.path.join(dest_dir, split, dtype)

            if not os.path.exists(src_path):
                continue

            for file in os.listdir(src_path):
                if file.startswith(partition_prefix):
                    shutil.copy2(os.path.join(src_path, file), os.path.join(dest_path, file))

    print(f"✅ Dataset prepared in '{dest_dir}' for partition {args.partition}")


if __name__ == "__main__":
    main()
