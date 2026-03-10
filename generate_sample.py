import torch
import torchvision
from PIL import Image
import os

def create_sample():
    # Load test dataset
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Get the very first image in the test set
    image_tensor, label = dataset[0] # which is a 7
    image_pil = torchvision.transforms.ToPILImage()(image_tensor)

    # Save to file
    output_path = f'sample_digit_{label}.png'
    image_pil.save(output_path)

    print(f"Created a sample test image from the dataset: '{output_path}'")
    print(f"The actual digit is: {label}")
    print(f"You can now test the model by running:")
    print(f"    python predict.py {output_path}")

if __name__ == "__main__":
    create_sample()
