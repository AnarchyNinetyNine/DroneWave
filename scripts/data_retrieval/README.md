# Phase 1: Data Retrieval and Auto-Annotation

## Introduction

Welcome to the first phase of the **DroneWave** project!  
This phase guides you through:

1. Requesting the UAV-Gesture dataset.
2. Downloading and unzipping videos.
3. Extracting frames at 25 FPS.
4. Automatically generating YOLOv12 bounding box annotations.
5. Uploading annotations to Roboflow for verification.

Following these steps ensures reproducibility and ease of use for researchers.

---

## Prerequisites

- Python 3.10+
- Access to the UAV-Gesture dataset (request via email)
- Installed dependencies (`environments/requirements.txt`)
- Roboflow account and API key
- ≥50GB free disk space for videos and frames

---

## Setup

1. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Directory Setup**:
   Ensure the `data/raw/` and `data/processed/` directories exist:
   ```bash
   mkdir -p data/raw data/annotations
   ```

## Steps

### 1. Requesting the Dataset

The UAV-Gesture dataset is available for academic research by contacting **Asanka Perera** at `asanka.perera@mymail.unisa.edu.au`. You may either utilize the provided script to streamline your request or draft a formal email independently to obtain the dataset.

**Script**: `scripts/data_retrieval/request_dataset.py`

```python
#!/usr/bin/env python3
"""
Request UAV-Gesture Dataset Script
----------------------------------
This script generates and sends a professional dataset request email to
Asanka Perera for academic research purposes. It prompts the user for
their credentials and displays the final email content before sending.
"""

import smtplib
from email.mime.text import MIMEText
from getpass import getpass

def create_email(your_email, your_name, institution, purpose):
    """Generate the MIMEText email for requesting the dataset."""
    message = (
        f"Dear Asanka Perera,\n\n"
        f"My name is {your_name} from {institution}. I am currently conducting research on "
        f"UAV gesture-based navigation systems. As part of this work, I would like to request "
        f"access to the UAV-Gesture dataset for academic research purposes.\n\n"
        f"Purpose of the work: {purpose}\n\n"
        f"Thank you very much for your consideration.\n\n"
        f"Best regards,\n{your_name}"
    )
    msg = MIMEText(message)
    msg['Subject'] = 'Request for UAV-Gesture Dataset'
    msg['From'] = your_email
    msg['To'] = 'asanka.perera@mymail.unisa.edu.au'
    return msg, message

def send_email(msg, your_email, password):
    """Send the prepared email using SMTP."""
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(your_email, password)
        server.send_message(msg)
    print("\n✅ Email sent successfully!")

if __name__ == "__main__":
    print("=== UAV-Gesture Dataset Request ===\n")

    # Prompt user input
    your_email = input("Enter your email address: ").strip()
    password = getpass("Enter your email password (input hidden): ")
    your_name = input("Enter your full name: ").strip()
    institution = input("Enter your institution name: ").strip()
    purpose = input("Briefly describe the purpose of your research: ").strip()

    # Create the email
    msg, final_message = create_email(your_email, your_name, institution, purpose)

    # Display final message for confirmation
    print("\n--- Email Preview ---")
    print(final_message)
    confirm = input("\nDo you want to send this email? (yes/no): ").strip().lower()

    if confirm == 'yes':
        send_email(msg, your_email, password)
    else:
        print("Email not sent. You can modify your inputs and try again.")
```

**Usage**:
```bash
python scripts/data_retrieval/request_dataset.py
```

**Instructions**:

1. The script will prompt you to enter your email, password, full name, institution, and purpose of the research.
2. It will display a styled preview of your email for confirmation before sending.
3. After confirmation, the email will be sent automatically to asanka.perera@mymail.unisa.edu.au.
4. Following your request, the provider will share a Google Drive link granting access to `UAVGesture.zip`, typically within a few business days.

### 2. Download & Unzip Dataset

Download `UAVGesture.zip` and place it in `data/raw/`. and unzipit. You may use the following script to automate the process.

**Script**: `scripts/data_retrieval/require_dataset.py`
This will downloads the dataset from Google Drive and extracts UAVGesture.zip into data/raw/.

**Usage**:
```bash
python scripts/data_retrieval/require_dataset.py
```

### 3. Auto-Annotation with YOLOv12

**Script**: `scripts/data_retrieval/annotate_dataset.py` 

This script:
1. Traverses all gesture folders inside data/raw/.
2. Extracts frames at 25 FPS from videos.
3. Performs YOLOv12 person detection.
4. Saves only the largest person bounding box per frame.
5. Converts frames to grayscale.
6. Saves frames and annotations under data/annotated/<gesture_folder>/<video_name>.

**Usage**:
```bash
python scripts/data_retrieval/annotate_dataset.py
```
Each gesture folder is assigned a unique class number automatically in the annotations.

### 4. Upload to Roboflow

This script uploads the extracted frames and annotations to your Roboflow account for manual verification; it also Includes retry mechanism in case of upload failures.

**Script**: `scripts/data_retrieval/roboflow_upload.py`

** Important**: Make sure to manually provide your Roboflow credentials and project info before running the script, `nano` the script and modify the following lines.

```python
ROBOFLOW_API_KEY = ""        # Enter your API key
PROJECT_NAME = ""            # Enter your Roboflow project name
WORKSPACE_NAME = ""          # Enter your Roboflow workspace
ANNOTATED_DIR = ""           # Path to your processed data, e.g., 'data/annotated'
MAX_RETRIES = 3
RETRY_DELAY = 5              # seconds

```
**Usage**:
```bash
python scripts/data_retrieval/roboflow_upload.py
```

## Next Steps

Proceed to [Phase 2: Data Processing and Manual Correction](scripts/data_processing/README.md) to verify annotations, clean the dataset, and prepare it for training.
