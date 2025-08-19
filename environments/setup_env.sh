#!/bin/bash
# -----------------------------------------------------------------------------
# setup_env.sh
# -----------------------------------------------------------------------------
# This script sets up a Python 3.10 virtual environment for the project and
# installs all required dependencies from requirements.txt.
#
# Steps:
# 1. Create a virtual environment in the '.venv' folder
# 2. Activate the virtual environment
# 3. Upgrade pip to the latest version
# 4. Install all dependencies listed in requirements.txt
#
# Usage:
#   bash setup_env.sh
# -----------------------------------------------------------------------------

# Step 1: Create a virtual environment named 'venv'
python3.10 -m venv .venv

# Step 2: Activate the virtual environment
source .venv/bin/activate

# Step 3: Upgrade pip to the latest version
pip install --upgrade pip

# Step 4: Install all required dependencies from requirements.txt
pip install -r requirements.txt

