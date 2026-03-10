import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps, ImageChops
import io
import base64
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

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

app = Flask(__name__)
CORS(app)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitCNN().to(device)

def load_loaded_model():
    try:
        if os.path.exists('digit_clf.pth'):
            model.load_state_dict(torch.load('digit_clf.pth', map_location=device, weights_only=True))
            model.eval()
            print("Model loaded successfully.")
        else:
            print("Model weights not found. Please wait for training to complete.")
    except Exception as e:
        print(f"Error loading model files: {e}")

from PIL import Image, ImageOps, ImageChops, ImageFilter

# Preprocessing to match the training transform (standard MNIST)
# Added a slight blur to match the anti-aliased nature of the training set
transform = transforms.Compose([
    transforms.Resize((28, 28), interpolation=Image.Resampling.LANCZOS),
    transforms.Grayscale(),
    lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.3)), # Soften edges
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST mean/std
])

def trim_and_center(image):
    # This helps mimic MNIST's centering
    # Find the bounding box of the non-black part (the digit)
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
        new_img = Image.new(image.mode, (new_size, new_size), (0,0,0,255) if image.mode == 'RGBA' else (0,))
        new_img.paste(cropped, (padding + (max_dim - w) // 2, padding + (max_dim - h) // 2))
        return new_img
    return image

@app.route('/')
def home():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Open image and ensure it has an alpha channel for composite later
        image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')

        # Create a solid black background
        background = Image.new('RGBA', image.size, (0, 0, 0, 255))
        
        # Paste the image on top of the black background
        alpha_composite = Image.alpha_composite(background, image)
        
        # Convert to Grayscale
        gray_image = alpha_composite.convert('L')
        
        # Auto-invert if image has a light background (e.g. photos of a digit on paper)
        img_np = np.array(gray_image)
        if np.mean(img_np) > 127:
            gray_image = ImageOps.invert(gray_image)

        # Robust Preprocessing: crop and center like MNIST
        centered_image = trim_and_center(gray_image)

        # Preprocess for the model
        input_tensor = transform(centered_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        probs_list = probabilities.tolist()
        prediction = int(torch.argmax(probabilities).item())
        word = DIGIT_WORDS[prediction]

        return jsonify({
            'prediction': prediction,
            'word': word,
            'probabilities': probs_list
        })
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_loaded_model()
    app.run(host='127.0.0.1', port=8080, debug=False)
