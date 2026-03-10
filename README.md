# Image Digit Classification Project 🎨🔢

A high-accuracy handwritten digit classification system featuring both a **Modern Web Application** and a **Terminal Interface**. Built with **PyTorch (CNN)** and **Flask**.

## 🌟 Key Features
- **CNN Architecture**: Custom Convolutional Neural Network trained on MNIST with 99%+ accuracy.
- **Dynamic Web App**: 
    - **Black Board**: Draw digits directly on a canvas with a mouse or touch screen.
    - **Photo Upload**: Upload your own handwritten digit photos.
    - **Live Probabilities**: Real-time bar charts showing confidence for each digit (0-9).
- **Centering Logic**: Advanced pre-processing that trims and centers your drawings to match the MNIST dataset format for maximum accuracy.
- **Terminal Interface**: Command-line tool for quick prediction on any image file.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Madhu-Varshini-Gurram/Image-Digit-Classification.git
   cd Image-Digit-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   ```bash
   python train.py
   ```

## 🚀 Usage

### Option 1: Web Interface (Recommended)
Launch the Flask server:
```bash
python app.py
```
Then visit: `http://127.0.0.1:8080`

### Option 2: Terminal Prediction
Test the model directly from your console:
```bash
python predict.py path/to/your_digit.png
```

## 🧠 Model Details
The system uses a **CNN** (Convolutional Neural Network) with:
- 2x Convolutional Layers + MaxPool
- Dropout for regularization
- 128-neuron Dense Layer
- 10-class Softmax output

Developed with 💖 for high-precision digit recognition.
