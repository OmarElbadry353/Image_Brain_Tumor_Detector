# Brain Tumor Detection CNN using PyTorch

This project contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying medical images to detect the presence of a brain tumor.

## Features
- **CNN Architecture**: A simple but effective CNN with two convolutional blocks.
- **Data Loading**: Uses `torchvision.datasets.ImageFolder` to load data from a structured directory.
- **Training & Validation**: Implements a standard training loop with a separate validation phase to monitor performance on unseen data.
- **Checkpointing**: Automatically saves the model with the best validation accuracy to `best_model.pth`.

## Project Structure
```
Image_Brain_Tumor_Detector/
├── brain_tumor_dataset/
│   ├── no/
│   │   ├── 1 no.jpeg
│   │   └── ...
│   └── yes/
│       ├── Y1.jpg
│       └── ...
├── main.py
└── README.md
```

## Dataset
The model expects the image data to be in the `brain_tumor_dataset` directory. This directory should contain two subdirectories:
- `yes`: Containing images of brains with tumors.
- `no`: Containing images of brains without tumors.

## Requirements
You will need Python 3 and the following libraries:
- `torch`
- `torchvision`

You can install them using pip:
```bash
pip install torch torchvision
```

## Usage
To start training the model, simply run the `main.py` script from your terminal:
```bash
python main.py
```
The script will start the training process, printing the loss and validation accuracy for each epoch.

## Output
- **Console Output**: The training progress will be printed to the console, showing the loss and accuracy for each epoch.
```
Using device: cuda
...
Epoch [1/10], Loss: 0.5811, Val Loss: 0.4987, Val Accuracy: 78.43%
New best model found! Saving to best_model.pth
...
Epoch [10/10], Loss: 0.0123, Val Loss: 0.0456, Val Accuracy: 98.04%
New best model found! Saving to best_model.pth
Finished Training
```
- **Model Checkpoint**: The script will save the best performing model to a file named `best_model.pth` in the project root directory. This file contains the model's weights, the optimizer's state, and the performance metrics at the time of saving.
