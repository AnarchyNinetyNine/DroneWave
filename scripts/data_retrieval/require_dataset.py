#!/usr/bin/env python3

"""
require_dataset.py
----------------
This script downloads the UAV-Gesture dataset from a Google Drive link
provided as a command-line argument, unzips it into a designated folder,
and prepares the directory structure for further processing.

Dependencies:
- gdown
- opencv-python (cv2)
- tqdm
- ultralytics (YOLO)
- scikit-learn (train_test_split)

Usage:
    python3 require_dataset.py <google_drive_file_id_or_url>
Example:
    python3 download_data.py 1a2b3c4d5e6f7g8h9i
"""

import os
import sys
import gdown
from zipfile import ZipFile

# -----------------------------
# Configuration
# -----------------------------
# Output path for the downloaded zip
output_zip = "data/raw/UAVGesture.zip"

# Directory to extract dataset
extract_dir = "data/raw"


# -----------------------------
# Download Dataset
# -----------------------------
def download_dataset(url: str, output: str):
    """
    Download a file from Google Drive using gdown.

    Args:
        url (str): Google Drive file URL or file ID.
        output (str): Path to save the downloaded file.
    """
    print(f"Downloading dataset from: {url}")
    gdown.download(url, output, quiet=False)
    print(f"Dataset downloaded to: {output}")


# -----------------------------
# Extract Dataset
# -----------------------------
def unzip_dataset(zip_path: str, extract_to: str):
    """
    Extract a zip file to the specified directory.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory to extract files to.
    """
    print(f"Extracting {zip_path} to {extract_to} ...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Check if Google Drive link/ID is provided
    if len(sys.argv) < 2:
        print("❌ Error: Please provide the Google Drive file URL or ID as a command-line argument.")
        print("Usage: python3 require_dataset.py <google_drive_file_id_or_url>")
        sys.exit(1)

    # Get URL or file ID from command-line
    url = sys.argv[1]

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Step 1: Download the dataset
    download_dataset(url, output_zip)

    # Step 2: Extract the downloaded zip
    unzip_dataset(output_zip, extract_dir)

    print("✅ Dataset is ready in 'data/raw'.")
