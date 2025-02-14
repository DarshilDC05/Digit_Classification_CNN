import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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

model = CNN()
model.load_state_dict(torch.load("./mnist_cnn_v1.pth", weights_only=True))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)  
])

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # adding a dimension (the batch one)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f"Predicted Digit: {prediction}")
    plt.axis("off")
    plt.show()

    return prediction

image_path = "./zero.png"
predicted_digit = predict(image_path)
print(f"Model Prediction: {predicted_digit}")