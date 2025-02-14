import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Normalization of data while importing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

# Importing data
train_dataset = datasets.MNIST(root="./mnist", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./mnist", train=False, transform=transform, download=True)

# Visualizing the dataset
# num_images = 100
# fig, axes = plt.subplots(10, 10)

# for i in range(num_images):
#     image, label = test_dataset[i]
#     image = image.numpy().squeeze()
#     row, col = divmod(i, 10)
#     axes[row, col].imshow(image, cmap="gray")
#     axes[row, col].set_title(f"{label}")
#     axes[row, col].axis("off")
# plt.show()

# Define data loaders
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0)


# Define CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
    
# initialize the model
model = CNN()

# move to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
    
# Cross-entropy loss and adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(loss = loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("Training Complete!")

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# saving the model
model_path = "./mnist_cnn_v1.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")