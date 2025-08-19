#!/usr/bin/env python3

"""
annotate_dataset.py
-----------------
This script traverses all subfolders inside 'data/raw', processes each video,
extracts frames, performs YOLOv12 person detection, saves only the largest
person bounding box per frame, converts frames to grayscale, and saves
frames and annotations in a structured directory under 'data/annotated'.

Each gesture folder is assigned a unique class number automatically.

Dependencies:
- opencv-python (cv2)
- tqdm
- ultralytics (YOLOv12)
- os, pathlib
"""

import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
raw_data_dir = os.path.join(os.getcwd(), 'data/raw')          # Input folder containing gesture folders
output_base_dir = os.path.join(os.getcwd(), 'data/annotated')  # Base output folder

os.makedirs(output_base_dir, exist_ok=True)

# Load YOLO model (YOLOv12, small model for demonstration)
model = YOLO('yolo12n.pt')

# -----------------------------
# Map each gesture folder to a unique class number
# -----------------------------
gesture_folders = sorted([f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))])
gesture_to_class = {gesture: idx for idx, gesture in enumerate(gesture_folders)}

# -----------------------------
# Helper function to process a single video
# -----------------------------
def process_video(video_path, output_dir, model, class_id):
    """
    Extract frames from a video, annotate largest person box, convert frames
    to grayscale, and save frames and annotations.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save frames and annotations.
        model (YOLO): Preloaded YOLO model.
        class_id (int): Class number to write in annotation file.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    annotations_dir = os.path.join(video_output_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- Step 1: YOLO Person Detection ----
        results = model(frame, verbose=False)
        person_boxes_xywhn = []

        for r in results:
            for cls_idx, xywhn in zip(r.boxes.cls, r.boxes.xywhn):
                class_name = model.names[int(cls_idx)]
                if class_name == "person":
                    person_boxes_xywhn.append(xywhn)

        # Save largest bounding box if detected
        if person_boxes_xywhn:
            areas = [(box[2] * box[3]).item() for box in person_boxes_xywhn]
            max_idx = areas.index(max(areas))
            best_box = person_boxes_xywhn[max_idx]
            x_center, y_center, width, height = best_box.tolist()

            annotation_filepath = os.path.join(
                annotations_dir, f"{video_name}_frame_{frame_count:04d}.txt"
            )
            with open(annotation_filepath, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # ---- Step 2: Convert frame to grayscale and save ----
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_filepath = os.path.join(
            video_output_dir, f"{video_name}_frame_{frame_count:04d}.jpg"
        )
        cv2.imwrite(frame_filepath, gray_frame)

        frame_count += 1

    cap.release()


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Traverse all subfolders in raw_data_dir
    for gesture_folder, class_id in gesture_to_class.items():
        gesture_path = os.path.join(raw_data_dir, gesture_folder)
        print(f"\nProcessing gesture folder: {gesture_folder} (Class ID: {class_id})")

        # Output folder for this gesture
        gesture_output_dir = os.path.join(output_base_dir, gesture_folder)
        os.makedirs(gesture_output_dir, exist_ok=True)

        # Process all video files in this folder
        for video_file in tqdm(os.listdir(gesture_path), desc=f"Videos in {gesture_folder}"):
            video_path = os.path.join(gesture_path, video_file)
            if not os.path.isfile(video_path) or not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue

            process_video(video_path, gesture_output_dir, model, class_id)

    print("\n✅ All videos processed and saved in 'data/annotated'.")

