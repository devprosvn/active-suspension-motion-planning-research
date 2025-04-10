<div align="center">
  <a href="https://utc2.edu.vn">
    <img src="https://raw.githubusercontent.com/devprostudio/devprostudio.github.io/main/utc2_logo.svg" alt="UTC2 Logo" width="200" height="200" style="margin-right: 20px">
  </a>
  <a href="https://fb.com/work.devpros">
    <img src="https://raw.githubusercontent.com/devprostudio/devprostudio.github.io/main/devpros_logo.svg" alt="DevPros Logo" width="200" height="200" style="margin-left: 20px">
  </a>
</div>

<div align="center">
  <h2>DevPros Team - Trường Đại học Giao thông Vận tải Phân hiệu tại TPHCM</h2>
  <h2>DevPros Team - University of Transport and Communications - Campus in Ho Chi Minh City</h2>
</div>

# Active Suspension Motion Planning Research
Nghiên cứu cải thiện hệ thống treo chủ động bằng Motion Planning để tối ưu độ ổn định xe tự hành.

## Giới thiệu

Dự án ứng dụng Motion Planning kết hợp AI để:
- Nhận diện địa hình bằng MobileViT + Depth Estimation
- Dự đoán tác động địa hình bằng Temporal Fusion Transformer (TFT)
- Điều khiển thời gian thực bằng Soft Actor-Critic (SAC)
- Mô phỏng và kiểm thử trên Carla 0.9.14

## Hướng phát triển

1. Hoàn thiện mô hình nhận diện địa hình:
   - Cải tiến MobileViT + Depth Estimation
   - Sử dụng tập dữ liệu lớn hơn từ Carla

2. Tối ưu hóa mô hình dự đoán dao động xe:
   - Cải thiện Temporal Fusion Transformer (TFT)
   - Tích hợp dữ liệu từ cảm biến IMU

3. Cải tiến hệ thống treo thích ứng:
   - Tăng cường thuật toán Soft Actor-Critic (SAC)
   - Tối ưu hóa CasADi + MPC

4. Nâng cao hiệu suất mô phỏng:
   - Sử dụng Carla Vehicle Physics API
   - Thiết lập kịch bản lái xe tự động

5. Hướng đến ứng dụng thực tiễn

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
