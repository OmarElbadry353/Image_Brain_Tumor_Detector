from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)

# 2. Model Definition
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorCNN()
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_file = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream).convert('RGB')

            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_file = img_str

            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)
                predicted = torch.sigmoid(output).item() > 0.5
                prediction = 'Tumor Detected' if predicted else 'No Tumor Detected'
    return render_template('index.html', prediction=prediction, image_file=image_file)