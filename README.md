<h1 align="center"> A Vehicular-Edge Federated, Quantized YOLOv12 System for Real-Time 3D Hand-Gesture–Based UAV Control </h1>

<div align="center">

![Version](https://img.shields.io/badge/version-v1.0.0-blue) 
![Release Date](https://img.shields.io/badge/release-August%2019%2C%202025-green)
![GitHub Stars](https://img.shields.io/github/stars/AnarchyNinetyNine/DroneWave?style=social)  

![Contributors](https://img.shields.io/badge/contributors-Ismail%20Lamaakal%2C%20Idris%20Elgarrab%2C%20Abdennour%20Abouach-orange)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

</div>
  

This repository is actively maintained. Check the [GitHub Releases](https://github.com/AnarchyNinetyNine/DroneWave/releases) for the latest updates, and consider starring the repository to stay informed!

## Introduction

**DroneWave** is an open-source project that enables intuitive control of unmanned aerial vehicles (UAVs) through human gestures, using advanced object detection and privacy-preserving federated learning. Built around the [UAV-Gesture dataset](https://asankagp.github.io/uavgesture/) and the YOLOv12 model, this project provides a clear pipeline to acquire, annotate, and process gesture data, then train an efficient, quantized model optimized for resource-constrained devices. Using the [Flower framework](https://flower.ai/), it supports distributed training across multiple devices while keeping data private. Designed for researchers, students, and developers, DroneWave offers a reproducible framework for exploring UAV navigation, computer vision, and distributed machine learning.  

We also leverage [Microsoft AirSim](https://github.com/microsoft/AirSim) to **simulate UAV flight and gesture-based control scenarios**, allowing safe, flexible experimentation before real-world deployment.  

The project is divided into three phases:

1. **Data Retrieval and Auto-Annotation**: Acquiring the UAV-Gesture dataset and generating initial annotations using YOLOv12.
2. **Data Processing and Manual Correction**: Uploading data to [Roboflow](https://roboflow.com/) for manual quality checking, cleaning, augmentation, and exporting in YOLOv12-compatible format.
3. **Federated Learning Setup**: Partitioning data across clients and training a federated, quantized model using Flower.
4. **Gesture Simulation in AirSim**: Using Microsoft AirSim to simulate UAVs, validate gesture-based control in a virtual environment, and bridge the gap between dataset-driven training and real-world UAV applications

Each phase is accompanied by a dedicated README file with detailed instructions, scripts, and dependencies.

---

## Project Structure

The repository is organized into directories corresponding to each phase of the project, ensuring modularity and ease of navigation. Below is the directory structure:

```
DroneWave/
├── data/
│   ├── raw/
│   └── annotations/
├── scripts/
│   ├── data_retrieval/
│   │   ├── request_dataset.py
│   │   ├── require_dataset.py
│   │   ├── annotate_dataset.py
│   │   └── README.md          ← (Phase 1 instructions)
│   └── data_processing/
│       ├── upload_to_roboflow.py
│       └── README.md          ← (Phase 2 instructions)
├── federated-learning/
│   ├── utils
│   │   ├── load_model.sh
│   │   └── load_dataset.py
│   ├── federated_learning/
│   │   ├── client_app.py
│   │   ├── server_app.py
│   │   ├── task.py
│   │   └── __init__.py
│   ├── pyproject.toml
│   └── README.md              ← (Phase 3 instructions)
├── simulation/
│   └── README.md              ← (Phase 4 instructions)
├── environments/
│   ├── requirements.txt
│   ├── setup_env.sh
│   └── setup_env.bat
└── README.md                  ← (Main overview)
```

### 📂 Directory Descriptions

- **data/**: Stores dataset files at different stages.  
  - **raw/**: Contains the unzipped UAV-Gesture dataset (videos).  
  - **annotations/**: Holds extracted frames, processed images, and annotations.  

- **scripts/**: Contains Python scripts for early phases, organized by task.  
  - **data_retrieval/**: Scripts for dataset acquisition, extraction, and auto-annotation with YOLOv12.  
  - **data_processing/**: Script for preparing data (uploading to Roboflow, cleaning, augmenting, and exporting in YOLOv12 format).  

- **federated-learning/**: Contains the full federated learning setup.  
  - **utils/load_model.sh, utils/load_dataset.py**: Utilities for loading models and datasets.  
  - **federated_learning/**: Flower apps for simulating federated learning (client, server, task management).  
  - **pyproject.toml**: Python project configuration (dependencies, build system).  

- **environments/**: Environment setup scripts and requirements.  
  - **requirements.txt**: Python dependencies.
  - **setup_env.sh / setup_env.bat**: Scripts to create and configure the Python environment (Linux/Mac & Windows).  

- **README.md**: Main project overview and quickstart guide.  

---

## 🚀 Prerequisites

To run this project, ensure you have:  
- **Python 3.10** or higher
- A [Roboflow](https://roboflow.com/) account for dataset management  
- A machine with a **GPU** (recommended for YOLOv12 training)  
- **Git** for cloning the repository  
- Basic familiarity with **command-line interfaces** and **Python scripting**  


## Quickstart Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AnarchyNinetyNine/DroneWave.git
   cd DroneWave
   ```

2. **Set Up the Virtual Environment**:
   ```bash
   bash environments/setup_env.sh
   ```
   _(Windows users can run **environments/setup_env.bat** instead)_

3. **Follow Phase-Specific Instructions**:
   - [Phase 1: Data Retrieval and Auto-Annotation](scripts/data_retrieval/README.md)
   - [Phase 2: Data Processing and Manual Correction](scripts/data_processing/README.md)
   - [Phase 3: Federated Learning Setup](federated-learning/README.md)
   - [Phase 4: Drone Control Simulation using Microsoft AirSim](simulation/README.md)

## How to Contribute

Contributions are always welcome!
To contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request (PR)

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Citation

If you use this project or the UAV-Gesture dataset in your research, please cite the original dataset paper:

> Perera, A. G., Law, Y. W., & Chahl, J. (2019). UAV-GESTURE: A Dataset for UAV Control and Gesture Recognition. [Paper link](https://asankagp.github.io/uavgesture/)

Additionally, please cite this repository:

@ARTICLE{11313047,
  author={Lamaakal, Ismail and Elgarrab, Idris and Alouach, Abdennour and Maleh, Yassine and El Makkaoui, Khalid and Ouahbi, Ibrahim and Alanezi, Ahmad and Khalifa, Hany S.},
  journal={IEEE Access}, 
  title={A Vehicular-Edge Federated, Quantized YOLOv12 System for Real-Time 3D Hand-Gestures- Based AAV Control}, 
  year={2026},
  volume={14},
  pages={3359-3385},
  doi={10.1109/ACCESS.2025.3647565}} [Paper link](https://ieeexplore.ieee.org/document/11313047))

## Contact

For questions or contributions, please open an issue on GitHub or contact [ismail.lamaakal@ieee.org].
