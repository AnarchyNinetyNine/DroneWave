@echo off
REM -----------------------------------------------------------------------------
REM setup_env.bat
REM -----------------------------------------------------------------------------
REM This script sets up a Python 3.10 virtual environment for the project and
REM installs all required dependencies from requirements.txt on Windows.
REM
REM Steps:
REM 1. Create a virtual environment in the '.venv' folder
REM 2. Activate the virtual environment
REM 3. Upgrade pip to the latest version
REM 4. Install all dependencies listed in requirements.txt
REM
REM Usage:
REM   setup_env.bat
REM -----------------------------------------------------------------------------

REM Step 1: Create a virtual environment named '.venv'
python -m venv .venv

REM Step 2: Activate the virtual environment
call .venv\Scripts\activate.bat

REM Step 3: Upgrade pip to the latest version
python -m pip install --upgrade pip

REM Step 4: Install all required dependencies from requirements.txt
pip install -r requirements.txt
