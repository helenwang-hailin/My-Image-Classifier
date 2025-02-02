import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json
from train import build_model

# Define arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Predict flower type from an image')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to saved checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

# Load the model from checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_model(arch=checkpoint['arch'], hidden_units=checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Process the image
def process_image(image_path):
    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

# Predict the class
def predict(image_path, model, top_k, device, category_names=None):
    model.to(device)
    image = process_image(image_path)
    image = image.to(device)
    
    with torch.no_grad():
        model.eval()
        outputs = model(image)
        probs, indices = torch.exp(outputs).topk(top_k)
        
    probs = probs.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(idx)] for idx in indices]
    else:
        class_names = indices

    return probs, class_names

def main():
    args = parse_args()
    
    # Check for GPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_checkpoint(args.checkpoint)
    
    # Predict
    probs, class_names = predict(args.image_path, model, args.top_k, device, args.category_names)
    
    # Print the result
    for i in range(args.top_k):
        print(f"{class_names[i]} with a probability of {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()