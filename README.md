# SolarDetect with YOLO-World ☀️

Fine-tuned **YOLO-World** (Vision-Language Model) for solar panel detection in satellite imagery, with a Gradio web interface for interactive inference.

---

## Overview

This project fine-tunes YOLO-World on a custom solar panel dataset to enable accurate detection in aerial/satellite images. Unlike standard YOLO models, YOLO-World leverages open-vocabulary detection, allowing zero-shot generalization before fine-tuning.

**Key features:**
- Fine-tuned YOLO-World on a custom `solar_dataset`
- Interactive Gradio web UI (`app.py`) for drag-and-drop inference
- Fully containerized with Docker for reproducible deployment
- Training pipeline with configurable YAML config (`solar.yaml`)

---

## Demo
<img width="1715" height="847" alt="image" src="https://github.com/user-attachments/assets/d9bfbb9c-f338-43a3-be7c-764aa71ce37f" />
link video: https://www.youtube.com/watch?v=R8qt4y3Bn4c

---

Run locally via Docker:

```bash
docker build -t solar-detection-app .
docker run -d -p 7860:7860 --name solar-container solar-detection-app
```

Then open: [http://localhost:7860](http://localhost:7860)

---

## Project Structure

```
SolarDetect-with-Yolo-world/
├── app.py              # Gradio inference UI
├── train.py            # Fine-tuning script
├── data.py             # Dataset preparation
├── solar.yaml          # YOLO training config
├── solar_dataset/      # Training data (images + labels)
├── runs/               # Training outputs & weights
├── Dockerfile
└── requirements.txt
```

---

## Quickstart

### Option A — Docker (recommended)

1. Clone and download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1ON2cr3DAURZVpPMPe4LUBs12jibinuiq/view?usp=sharing), place the `.pt` file in the project root.

2. Build and run:
```bash
docker build -t solar-detection-app .
docker run -d -p 7860:7860 --name solar-container solar-detection-app
```

### Option B — Local (Python)

```bash
git clone https://github.com/RayWuttinun/SolarDetect-with-Yolo-world.git
cd SolarDetect-with-Yolo-world
pip install -r requirements.txt
python app.py
```

---

## Training from Scratch

```bash
python train.py
```

After training, copy the best weights from `runs/detect/solar_detection/yolo_world_tuning/weights/best.pt` to the project root, then launch the app.

---

## Tech Stack

| Component | Technology |
|---|---|
| Detection model | YOLO-World (fine-tuned) |
| Training framework | Ultralytics |
| Web UI | Gradio |
| Containerization | Docker |
| Language | Python 3.10+ |

---

## Dataset

Custom solar panel dataset (`solar_dataset/`) with YOLO-format annotations. Training config defined in `solar.yaml`.

---
