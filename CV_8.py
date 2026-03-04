import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import requests
import cv2
import numpy as np

# Load model
model = models.resnet50(pretrained=True)
model.eval()

# Load labels
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
labels = requests.get(LABELS_URL).json()

# Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = Image.open(image_path).convert('RGB')

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(input_tensor)

prob = torch.nn.functional.softmax(output[0], dim=0)
top_prob, top_catid = torch.topk(prob, 1)

label = labels[str(top_catid.item())][1]
confidence = top_prob.item() * 100

print(f"Prediction: {label} ({confidence:.2f}%)")