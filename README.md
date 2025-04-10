# active-suspension-motion-planning-research
UTC2 Research about Active Suspension with Motion Planning. Project's done by DevPros.

## Terrain Recognition and Depth Estimation in CARLA

This project implements a terrain recognition and depth estimation system using MobileViT architecture in the CARLA 0.9.14 simulation environment, specifically on the Town10 map.

## Project Overview

The system is designed to:
1. Recognize different terrain types (asphalt, gravel, sand, snow, bumpy roads) from camera images
2. Estimate depth from the same images
3. Operate in real-time within the CARLA simulation environment

## Project Structure

```
.
├── data_collection/       # Scripts for collecting training data from CARLA
├── data_processing/       # Scripts for preprocessing and preparing datasets
├── models/                # Model architecture definitions
├── training/              # Training scripts and utilities
├── evaluation/            # Model evaluation scripts
├── inference/             # Real-time inference in CARLA
└── utils/                 # Utility functions and helpers
```

## Requirements

- Python 3.7.9
- Pip 20.1.1
- CARLA 0.9.14
- PyTorch 1.13.1
- Other dependencies listed in requirements.txt

## Setup and Installation

1. Ensure CARLA 0.9.14 is installed
2. Set up a Python 3.7.9 environment
3. Install dependencies: `pip install -r requirements.txt`
