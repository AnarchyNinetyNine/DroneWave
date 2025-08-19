#!/usr/bin/env python3
"""
upload_to_roboflow.py
---------------------
This script uploads processed UAV-Gesture frames and their YOLOv12 annotations
to a Roboflow project. It traverses the 'data/annotated' folder structure,
matching frames to their annotations. Failed uploads are retried with a delay.

⚠️ USER CONFIGURATION REQUIRED:
- Set your Roboflow API key
- Set your Roboflow project name
- Set the path to your annotated dataset

Dependencies:
- roboflow
- os
- time
- logging
"""

import os
import time
import logging
from roboflow import Roboflow

# -----------------------------
# USER CONFIGURATION
# -----------------------------
# TODO: Provide your Roboflow API key here
ROBOFLOW_API_KEY = ""  # e.g., "YOUR_API_KEY"

# TODO: Provide your Roboflow project name here
PROJECT_NAME = ""  # e.g., "uav_gesture"

# TODO: Provide the path to your annotated dataset
ANNOTATED_DIR = ""  # e.g., os.path.join(os.getcwd(), "data/annotated")

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# Initialize Roboflow
# -----------------------------
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(PROJECT_NAME)

# -----------------------------
# Upload Function
# -----------------------------
def upload_image(image_path, annotation_path):
    """
    Upload an image and its annotation to Roboflow with retries on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            project.upload(image_path=image_path, annotation_path=annotation_path)
            logging.info(f"✅ Uploaded: {image_path}")
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed for {image_path}: {e}")
            if attempt < MAX_RETRIES:
                logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"Failed to upload {image_path} after {MAX_RETRIES} attempts.")
                return False

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    for gesture_folder in os.listdir(ANNOTATED_DIR):
        gesture_path = os.path.join(ANNOTATED_DIR, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        logging.info(f"Processing gesture folder: {gesture_folder}")

        for video_folder in os.listdir(gesture_path):
            video_path = os.path.join(gesture_path, video_folder)
            if not os.path.isdir(video_path) or video_folder == 'annotations':
                continue

            annotations_path = os.path.join(video_path, 'annotations')
            if not os.path.isdir(annotations_path):
                logging.warning(f"No annotations folder for {video_path}. Skipping.")
                continue

            for filename in os.listdir(video_path):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(video_path, filename)
                    annotation_filename = filename.replace('.jpg', '.txt')
                    annotation_file_path = os.path.join(annotations_path, annotation_filename)

                    if os.path.exists(annotation_file_path):
                        upload_image(image_path, annotation_file_path)
                    else:
                        logging.warning(f"No annotation found for {image_path}. Skipping upload.")
