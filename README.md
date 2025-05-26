# MAD-CNN for Robot Collision Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

A deep learning-based system for robotic collision detection using temporal signal processing and attention mechanisms.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)


## Features
- MAD-CNN architecture for temporal signal processing
- Multi-stiffness level collision detection

## Requirements
- Python 3.10
- Conda (Miniconda/Anaconda)
- NVIDIA GPU (Optional but recommended)

## Installation
```bash
# git clone https://github.com/yourusername/collision-detection.git
cd MAD_CNN
conda env create -f environment.yml
```

## Project Structure
collision-detection/
├── config.py              # Central configuration parameters
├── helpers.py             # Data loading, model helpers
├── main.py                # Main training/evaluation script
├── network_built.py       # Model architecture definitions
├── evaluation_CD.py       # Evaluation metrics and visualization
├── Data_process.py        # Data preprocessing utilities
├── environment.yml        # Conda environment specification
├── README.md              # This document
└── requirements.txt       # Requirements

## Usage
python main.py


