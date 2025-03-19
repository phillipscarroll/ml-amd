# Torch-DirectML

Official MS Documentation: <a href="https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows">Enable PyTorch with DirectML on Windows</a>

It looks like we may be able to test some things at near native speeds using the RX 9070 XT in PyTorch via DirectML. This should technically work but I have never used DirectML for PyTorch models/training.

```
# Setup the environment
conda create -n directml python=3.12 -y
conda activate directml

# Make Jupyter folder
mkdir Jupyter-DirectML
cd Jupyter-DirectML

# Install Torch-DirectML
pip install torch-directml jupyterlab

# Start a Jupyter Lab
jupyter lab password
jupyter lab --ip 10.0.0.35 --port 8888 --no-browser
```

Connect via VSCode or open the url in the console output. Start a notebook and run the following:

```
import torch
import torch_directml
dml = torch_directml.device()
print(dml)
```

This 100% locked up my workstation, testing just the `import torch` module:

```
import torch
```

Great success.

Adding the `import torch_directml` module:

```
import torch
import torch_directml
```

Great success.

Setting a var to the device `dml = torch_directml.device()`

```
import torch
import torch_directml
dml = torch_directml.device()
```

Great success.

Ok everthing seems to be working, this might have been a fluke.

Trying to print what device is contained in the var dml `print(dml)`:

```
import torch
import torch_directml
dml = torch_directml.device()
print(dml)
```

This works and the output is: `privateuseone:0`

Let's test this on one of my existing ipynb's. This will require some packages to be installed and some tweaks to the code.

Install matplotlib, tqdm: `pip install matplotlib tqdm`

This is working but its only offloading to the GPU a small amount. The GPU will load from 3% to 15-17% but the cpu will go to 65-75% so Im not sure what is happening. This will require some research as this is my literal first time trying out DirectML in PyTorch.

Thoughts: Drivers, Pytorch config, maybe we need some DirectML SDK features. It could be that the 9070 XT is to new and just isnt supported in this capacity, DirectML definitely works for Stable Diffusion via Amuse so this leads me to think there is an issue feeding the GPU with/or configs.

In dataload these options reduced memory footprint and cpu load but also reduced gpu load.
`pin_memory=False, num_workers=16`

Looks like I'm loading data to ram and not vram.

```
import torch
import torch_directml
import torch.nn as nn
from torch.amp import GradScaler, autocast
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import time
import tqdm

hp = {
    "batch_size": 32,
    "epochs": 25,
    "random_seed": 42,
    "randomize_seed": True,
    "cpu_only": False,
    "device": "privateuseone:0",
}

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Randomize seed if set to True
if hp['randomize_seed']:
    hp['random_seed'] = random.randint(0, 1000000000)
logging.info(f"Seed set to: {hp['random_seed']}")  

# Simple CNN Class
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Device configuration
def get_device():
    """
    This will check for an Intel XPU or CUDA device and return it if available, otherwise it will return CPU.

    Returns the torch device to use.
    """
    return "privateuseone:0"

def train_model(epochs, model, train_loader, device, optimizer, criterion, scaler=None):

    # Start timer
    start_time = time.time()

    # 5. Training the Model
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{hp["epochs"]}], Loss: {running_loss/len(train_loader):.4f}')

        # End timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

# 7. Visualizing Some Predictions
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Main function
def main():

    # 2 Dataset, Dataloader, Transform
    # The transform using (0.5, 0.5, 0.5) is used to normalize the image data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Download and load the training data
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    # Download and load the test data
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    # Create the dataloader for training and testing data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=hp['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                             batch_size=hp['batch_size'], shuffle=False)
    # 3 SimpleCNN Class
    model_0 = SimpleCNN().to(hp["device"])
    
    # 4 Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_0.parameters(), lr=0.001)

    train_model(hp["epochs"], model_0, train_loader, hp["device"], optimizer, criterion)

    # 6. Evaluating the Model
    model_0.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(hp["device"]), labels.to(hp["device"])
            outputs = model_0(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')

    # Get random test images and predictions
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(hp["device"]), labels.to(hp["device"])

    # Display images
    imshow(torchvision.utils.make_grid(images.cpu()))
    print('GroundTruth:', ' '.join(f'{train_dataset.classes[labels[j]]}' for j in range(4)))

    # Predict and display results
    outputs = model_0(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:', ' '.join(f'{train_dataset.classes[predicted[j]]}' for j in range(4)))

    # 8. Saving the Model
    torch.save(model_0.state_dict(), 'cnn_cifar10.pth')
    print("Model saved as cnn_cifar10.pth")

# Run the main function
if __name__ == '__main__':
    main()
```