#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# load_model.sh
# -----------------------------------------------------------------------------
# This script downloads the YOLOv12n pre-trained model from Ultralytics.
# The model will be saved in the previous directory.
# -----------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# Model URL
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt"

# Destination file
DEST_FILE="../yolo12n.pt"

# Check if the model already exists
if [ -f "$DEST_FILE" ]; then
    echo "✅ Model already exists: $DEST_FILE"
else
    echo "⬇️  Downloading YOLOv12n model..."
    curl -L "$MODEL_URL" -o "$DEST_FILE"
    echo "✅ Download complete: $DEST_FILE"
fi
