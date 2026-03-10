import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageChops
import sys
import os
import numpy as np

# Improved CNN Model for higher accuracy and robustness (must match train.py)
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 7x7
            nn.Dropout2d(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def trim_and_center(image):
    # This helps mimic MNIST's centering
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        # Crop to the digit
        cropped = image.crop(bbox)
        # Pad it back into a square with some padding (approx 20% like MNIST)
        w, h = cropped.size
        max_dim = max(w, h)
        padding = int(max_dim * 0.2)
        new_size = max_dim + 2 * padding
        
        # Create a new black square image and paste the cropped digit onto it (centered)
        new_img = Image.new(image.mode, (new_size, new_size), (0,))
        new_img.paste(cropped, (padding + (max_dim - w) // 2, padding + (max_dim - h) // 2))
        return new_img
    return image

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitCNN().to(device)
    
    try:
        model.load_state_dict(torch.load('digit_clf.pth', map_location=device, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    from PIL import Image, ImageOps, ImageChops, ImageFilter
    transform = transforms.Compose([
        transforms.Resize((28, 28), interpolation=Image.Resampling.LANCZOS),
        transforms.Grayscale(),
        # Soften to match MNIST dataset
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.4)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        image = Image.open(image_path).convert('L')
        # Robust Preprocessing: crop and center like MNIST
        if np.mean(np.array(image)) > 127:
            image = ImageOps.invert(image)
        
        centered_image = trim_and_center(image)
        input_tensor = transform(centered_image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    probs_list = probabilities.tolist()
    prediction = int(torch.argmax(probabilities).item())

    # Formatted Output
    print("\n" + "="*45)
    print(f"IMAGE ANALYSIS : {image_path}")
    print("="*45)
    print("Probabilities per class (0-9):")
    for digit, prob in enumerate(probs_list):
        print(f"  Digit {digit}:  {prob*100:6.3f} %")
    
    print("-" * 45)
    print(f"TERMINAL PREDICTION: >> {prediction} <<")
    print("="*45 + "\n")

if __name__ == '__main__':
    main()
