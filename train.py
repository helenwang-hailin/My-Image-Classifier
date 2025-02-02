import argparse
import json

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from torchvision import datasets, models, transforms

# Define arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on the flower dataset')
    parser.add_argument('data_dir', type=str, help='Directory for flower data')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoint')
    parser.add_argument('--arch', type=str, choices=['vgg13', 'vgg16', 'resnet'], default='vgg13', help='Choose the architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

# Define a function to load the data
def load_data(data_dir):
    train_dir = Path(data_dir) / 'train'
    valid_dir = Path(data_dir) / 'valid'
    test_dir = Path(data_dir) / 'test'

    # Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform = training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    testing_data = datasets.ImageFolder(test_dir, transform = testing_transforms)

    # Define the dataloaders
    train_loader = DataLoader(training_data, batch_size = 64, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(validation_data, batch_size = 64, shuffle = True, num_workers = 2)
    test_loader = DataLoader(testing_data, batch_size = 64, shuffle = True)

    return train_loader, valid_loader, test_loader, training_data.class_to_idx

# Define a function to build the model
def build_model(arch='vgg13', hidden_units=512):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif arch == 'resnet':
        model = models.resnet18(pretrained=True)
        classifier = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 102),
            nn.LogSoftmax(dim=1)
        )
        model.fc = classifier

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

# Define the training function
def train_model(model, train_loader, valid_loader, epochs, learning_rate, device):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    print_every = 10  # Define the print interval

    for epoch in range(epochs):
        running_loss = 0
    
        # Training phase
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
        
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
         
            # Print progress every 'print_every' batches
            if batch_idx % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {batch_idx} | Loss: {loss.item():.4f}")

        # Validation phase (no gradient computation)
        model.eval()
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
            
                # Calculate accuracy
                ps = torch.exp(outputs)  # Convert log probabilities to probabilities
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Print results for the epoch
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {running_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {validation_loss/len(valid_loader):.4f} | Accuracy: {accuracy/len(valid_loader)*100:.2f}%\n")

    return model

# Save the checkpoint
def save_checkpoint(model, save_dir, arch, epochs, learning_rate, hidden_units, class_to_idx, optimizer):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'learning_rate': learning_rate,
        'class_to_idx': class_to_idx,  # Save class index mapping
        'classifier': model.classifier,  # Save classifier structure
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }

    torch.save(checkpoint, save_dir)
    print(f"Model saved to {save_dir}")

def main():
    args = parse_args()
    
    # Load data
    train_loader, valid_loader, test_loader, class_to_idx = load_data(args.data_dir)
    
    # Build model
    model = build_model(arch=args.arch, hidden_units=args.hidden_units)
    
    # Check for GPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Train the model
    model = train_model(model, train_loader, valid_loader, args.epochs, args.learning_rate, device)
    
    # Save the model
    save_checkpoint(model, args.save_dir, args.arch, args.epochs, args.learning_rate, args.hidden_units, class_to_idx, optim.Adam(model.classifier.parameters(), lr=args.learning_rate))

if __name__ == "__main__":
    main()
