import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Improved CNN Model for higher accuracy and robustness
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # Convolutional layers to catch spatial features (better for drawing)
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

def main():
    # Increased augmentation for real-world drawing robustness
    transform_train = transforms.Compose([
        transforms.RandomRotation(15), 
        transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading MNIST dataset with advanced augmentation...")
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for more epochs for high accuracy
    epochs = 15
    print(f"Starting training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Model Accuracy on Test Set: {100 * correct / total:.2f}%")

    # Save
    print("Saving model weight to digit_clf.pth...")
    torch.save(model.state_dict(), 'digit_clf.pth')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()
