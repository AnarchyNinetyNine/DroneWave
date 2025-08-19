# Phase 2: Data Processing and Roboflow Upload

## Introduction

In **Phase 2** of the **DroneWave** project, the goal is to take the annotated UAV-Gesture dataset generated in Phase 1, verify the annotations, and upload the processed frames to Roboflow. This ensures that all frames and corresponding YOLOv12 bounding box annotations are correctly structured and ready for further training or analysis.

This phase emphasizes:
- Traversing the structured `data/annotated` directory
- Matching frames to their annotations
- Reliable upload to a Roboflow project
- Automatic retries in case of upload failures

Proper configuration of your Roboflow credentials and dataset paths is required before execution.

---

## Prerequisites

- Completed **Phase 1**: dataset downloaded, unzipped, frames extracted, and YOLOv12 annotations generated
- Roboflow account with a valid API key
- Python 3.10+ and required dependencies installed (`roboflow`, `os`, `time`, `logging`)

---

## Steps

### 1. Configure Upload Script

Edit `scripts/data_processing/upload_to_roboflow.py`:

```python
# USER CONFIGURATION
ROBOFLOW_API_KEY = "<YOUR_API_KEY>"
PROJECT_NAME = "<YOUR_PROJECT_NAME>"
ANNOTATED_DIR = os.path.join(os.getcwd(), "data/annotated")
```

### 2. Run the Upload Script

Execute the script to traverse the annotated dataset and upload frames:

```bash
python scripts/data_processing/upload_to_roboflow.py
```
Behavior:

- Iterates over each gesture folder and video subfolder
- Matches .jpg frames to their .txt YOLO annotations
- Uploads each image to the specified Roboflow project
- Retries failed uploads up to 3 times with a 5-second delay
- Logs successful uploads and warnings for missing annotations

## Next Steps: Phase 3 – Federated Learning

Once the dataset is uploaded and verified in Roboflow:

1. Download verified dataset from Roboflow in YOLOv12 format.
2. Prepare local or cloud-based clients for federated learning (FL) experiments.
3. Implement the FL pipeline:
 - Each client trains locally on subsets of the UAV-Gesture dataset
 - A central server aggregates model updates
 - Optionally, evaluate global model performance on a held-out validation set
4. Integrate YOLOv12 model with the federated learning framework for gesture recognition.

[Phase 3: Federated Learning Setup](scripts/federated-learning/README.md) aims to leverage distributed model training for privacy-preserving and scalable UAV gesture recognition experiments.
