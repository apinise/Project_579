from PIL import Image
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

def predict_image(image_path, model, device, transform, class_names):
    """
    Predicts the class of an image using a trained model.

    Args:
        image_path (str): Path to the image.
        model (torch.nn.Module): Trained PyTorch model.
        device (torch.device): Device to run inference on.
        transform (torchvision.transforms.Compose): Transformations to apply to the image.
        class_names (list): List of class names (disease labels).

    Returns:
        str: Predicted class label.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Apply transformations
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move to device
    input_tensor = input_tensor.to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    
    # Get class label
    predicted_label = class_names[predicted_class.item()]
    return predicted_label

def calculate_test_accuracy(model, test_loader, device, classes):
    """
    Calculate the accuracy of the model on the entire test set.

    Parameters:
    - model: Trained PyTorch model.
    - test_loader: DataLoader for the test set.
    - device: Device on which the model is running.

    Returns:
    - accuracy: Test accuracy as a percentage.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Perform forward pass
            outputs = model(inputs)
            
            # Get predictions by taking the class with the highest score
            _, predicted = torch.max(outputs, 1)
            
            # Update the total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    return accuracy

def plot_confusion_matrix(model, data_loader, device, classes):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def classification_metrics(model, data_loader, device, classes):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    
dml = torch_directml.device(1)
dataset_path = './data/vipoooool/new-plant-diseases-dataset/versions/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
training_path = dataset_path + '/train'
testing_path = dataset_path + '/valid/test'
diseases = os.listdir(training_path)

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),    # Resize images to 256x256
    transforms.ToTensor(),            # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalization for ResNet
])

batch_size = 32
test_dataset = ImageFolder(root=testing_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Recreate the model architecture
loaded_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, len(diseases))  # Match number of classes
loaded_model = loaded_model.to(dml)  # Move to DirectML device

# Load the saved weights
model_path = "plant_disease_model.pth"
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()  # Set to evaluation mode
print("Model loaded successfully.")

# Call the function after testing
#classification_metrics(loaded_model, test_loader, dml, diseases)

#plot_confusion_matrix(loaded_model, test_loader, dml, diseases)

# Calculate test accuracy
test_accuracy = calculate_test_accuracy(loaded_model, test_loader, dml, diseases)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Path to your image
image_paths = [
    "infected_1_256x256.jpg",
    "infected_2_256x256.jpg",
    "normal_1_256x256.jpg",
]

predicted_labels = []

for image_path in image_paths:
    # Predict using the loaded model
    predicted_label = predict_image(image_path, loaded_model, dml, transform, diseases)
    predicted_labels.append(predicted_label)

# Plot images with predicted labels
fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))

for ax, image_path, label in zip(axes, image_paths, predicted_labels):
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_title(label)
    ax.axis("off")  # Hide axes

plt.tight_layout()
plt.show()