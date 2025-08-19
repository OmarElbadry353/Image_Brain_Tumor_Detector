---
title: Brain Tumor Detector
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: flask
app_file: app.py
---

# Brain Tumor Detector

This is a web application that uses a Convolutional Neural Network (CNN) to detect the presence of a brain tumor in MRI images. This project is built with PyTorch and Flask.

**How to use the web app:**
1.  Upload an MRI image of a brain.
2.  The model will predict whether a tumor is present or not.

## Model Training
The model was trained on the "Brain Tumor Dataset". The training script (`main.py`) uses a simple CNN architecture and saves the best performing model to `best_model.pth`.

## Requirements
- `torch`
- `torchvision`
- `Pillow`
- `flask`