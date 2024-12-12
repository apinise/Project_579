import kagglehub
import os
import numpy as np
import pandas as pd
import torch
import torch_directml
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchsummary import summary

# Download latest version (Already done + moved)
# path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset", "data")
# print("Path to dataset files:", path)

dataset_path = './data/vipoooool/new-plant-diseases-dataset/versions/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
training_path = dataset_path + '/train'
testing_path = dataset_path + '/valid/valid'
diseases = os.listdir(training_path)

dml = torch_directml.device(1)
print(torch_directml.device_name(1))

# Print disease names in dataset
print(diseases)
print("Total disease classes are: {}".format(len(diseases)))

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),    # Resize images to 256x256
    transforms.ToTensor(),            # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalization for ResNet
])

# Load datasets
train_dataset = ImageFolder(root=training_path, transform=transform)
test_dataset = ImageFolder(root=testing_path, transform=transform)

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define ResNet-18 model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(diseases))  # Adjust the final layer for 38 classes

# Move model to DirectML device
model = model.to(dml)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Example: Using SGD instead of Adam
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Testing Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Training and evaluation loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, train_loader, criterion, optimizer, dml)
    test(model, test_loader, criterion, dml)

# Save the model
model_path = "plant_disease_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
