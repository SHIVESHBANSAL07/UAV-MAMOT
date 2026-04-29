# Metadata-Assisted Multi-Object Tracking for UAV-Based Surveillance

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

## Project Abstract
This project implements a Metadata-Assisted Multi-Object Tracking (MA-MOT) system for UAV-based surveillance using YOLOv8 and OpenCV. The system fuses drone sensor data (GPS, altitude, camera yaw) with a Kalman filter tracker to reduce identity switches caused by camera motion. Trained on VisDrone2019-DET dataset achieving 87.8% mAP50 on car detection.


## Problem Statement
Tracking small objects in drone video is difficult due to:
- Constant camera movement causing position shifts
- Small object size at typical drone altitudes (80-120m)
- Objects temporarily leaving the camera frame
- Background noise from reflections and weather
- High identity switch rate in standard trackers

## Role of Edge Computing
- YOLOv8 inference runs on edge GPU without cloud dependency
- Fully offline operation suitable for remote surveillance
- Reduced latency: 2.2ms inference per frame
- Real-time tracking without internet connectivity

## System Pipeline
Input Video
→ Preprocessing (resize to 640x640)
→ YOLOv8 Detection (bounding boxes + confidence)
→ Metadata Fusion (GPS/altitude/yaw correction)
→ Kalman Filter Tracker (Hungarian algorithm)
→ Output Video (IDs + FPS + inference time)
## Model Details
| Property | YOLOv8n | YOLOv8s |
|----------|---------|---------|
| Classes | 10 (all VisDrone) | 1 (car only) |
| mAP50 | 0.267 | 0.878 |
| Inference Time | 2.2ms | ~5ms |
| Model Size | 6.2MB | 21.5MB |
| Training Epochs | 50 | 80 |
| Image Size | 640x640 | 1280x1280 |

## Training Details
- Dataset: VisDrone2019-DET (6,471 train images)
- Platform: Google Colab Tesla T4 GPU
- Framework: PyTorch + Ultralytics
- Optimizer: AdamW (lr=0.000714)

## Results
| Metric | Baseline | MA-MOT |
|--------|----------|--------|
| Avg Inference Time | 30.67ms | 153.62ms |
| Average FPS | 32.61 | 6.51 |
| ID Switches | 86 | 84 |
| mAP50 (cars) | - | 0.878 |
| Precision | - | 0.869 |
| Recall | - | 0.815 |

## Project Structure
UAV-MAMOT/
├── main.py
├── inference.py
├── tracker.py
├── utils.py
├── preprocessing.py
├── training.py
├── config.py
├── logger.py
├── requirements.txt
└── metadata/
└── drone_meta.json
## Setup Instructions
```bash
git clone https://github.com/SHIVESHBANSAL07/UAV-MAMOT.git
cd UAV-MAMOT
pip install -r requirements.txt
python main.py
```

## Submitted By
- **Name:** Shivesh Bansal
- **Roll Number:** 1024240047
- **Department:** Computer Science and Engineering
- **Institute:** Thapar Institute of Engineering and Technology, Patiala
- **Year:** April 2026