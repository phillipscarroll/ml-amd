# Archived instructions for setting up Navi31 (RX 7900 GRE/XT/XTX) for various ML services

This will mostly be out of date but I wanted a place to archive this setup from my old website/youtube channel.

These instructions are not streamlined and probably should not be followed

### Requirements

```
Windows 11

Radeon GPU: 7900 GRE, 7900 XT, or 7900 XTX. Other GPU's may work such as a 7800 XT but are not officially supported. I personally have tested this on both the 7900 GRE and 7900 XTX.

HIP SDK Driver: 6.1.0

Adrenalin Driver: 24.6.1 (I have also tested on latest drivers and it works fine)

ZLUDA

Miniconda

Git

WSL2
```

### Step 1: Install Adrenalin Driver: 24.6.1

```
https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html

If for some reason this version is not available, I have a copy here:

https://drive.google.com/file/d/1Evr9gzAsBxgb9VeJx-2KQ2A_cKkloV3k/view?usp=sharing

Install and reboot

NOTE: This is the only step I wont show on video because of recording issues while installing GPU drivers. Install this version, click next through everything, no changes are needed.
```

### Step 2: Install HIP SDK Driver: 6.1.0

```
https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html

If for some reason this version is not available, I have a copy here:

https://drive.google.com/ le/d/1nt5CQtTmC4uznOY2xN45pDAfW0fd7ZsP/view?usp=sharing

Install and reboot
```

### Step 3: Turn on WSL2

```
https://docs.microsoft.com/en-us/windows/wsl/install

Open powershell as administrator and run the following command: 

wsl --install

NOTE: If this fails you may need to enable virtualization in your BIOS. This is a
common issue with WSL2 on Intel.

Reboot after this is done
```

### Step 4: Install Miniconda

```
https://docs.conda.io/en/latest/miniconda.html

If for some reason this version is not available, I have a copy here:

https://drive.google.com/ le/d/1_SBpgwump9XZXiCzOql-zhf9Y2lPF3v8/view?usp=sharing

Click next through the installer, no changes are needed
```

### Step 5: Install Git

```
https://git-scm.com/download/win

If for some reason this version is not available I have a copy here:

https://drive.google.com/ le/d/1CfddS9h1n3LhyiN-s6nmngp0ZI-qOClp/view?usp=sharing

Click next through the installer, no changes are needed
```

### Step 6: Download a copy of ZLUDA

```
This project was active, then canceled and recently in Q4 2024 it has been restarted independent of AMD.

I am using ZLUDA version ZLUDA 3.8.3

You may download from the following link:

https://drive.google.com/ le/d/17_TprYhYro-v9ghAuG8p2rQXDBiGcLbi/view?usp=sharing

Unzip this to a location of your choice, I will be using:

F:\AI\ZLUDA

This will be important in a moment
```

### Step 7: Edit Windows environment variables

```
Add to the user path: C:\Program Files\Git\bin

Add to the sytem path:

%HIP_PATH%bin

F:\AI\ZLUDA
```

### Step 8: Create WSL2 Ubuntu 22.04 instance

```
Open powershell:

wsl --install --d Ubuntu-22.04

This will prompt you to enter a username and password for the new instance.

NOTE: You may want to read up on WSL2 and how to run multiple instances of a distro. Ideally you would create this image then clone it for future projects but what we are doing works just fine.
```

### Step 9: Configure Ubuntu 22.04

```
Once inside Ubuntu, enter these commands, I have updated this to be just 2 lines,
one short update and one very long chain of commands:

sudo apt update -y

wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb && sudo apt install ./amdgpu-install_6.1.60103-1_all.deb -y && sudo amdgpu-install --list-usecase && amdgpu-install -y --usecase=wsl,rocm --no-dkms && rocminfo && sudo apt install python3-pip -y && pip3 install --upgrade pip wheel && wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/torch-2.1.2%2Brocm6.1.3-cp310-cp310-linux_x86_64.whl && wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/torchvision-0.16.1%2Brocm6.1.3-cp310-cp310-linux_x86_64.whl && wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/pytorch_triton_rocm-2.1.0%2Brocm6.1.3.4d510c3a44-cp310-cp310-linux_x86_64.whl && pip3 uninstall torch torchvision pytorch-triton-rocm numpy && pip3 install torch-2.1.2+rocm6.1.3-cp310-cp310-linux_x86_64.whl torchvision-0.16.1+rocm6.1.3-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-2.1.0+rocm6.1.3.4d510c3a44-cp310-cp310-linux_x86_64.whl numpy==1.26.4 && location=`pip show torch | grep Location | awk -F ": " '{print $2}'` && cd ${location}/torch/lib/ && rm libhsa-runtime64.so* && cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so && python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure' && python3 -c 'import torch; print(torch.cuda.is_available())' && python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))" && python3 -m torch.utils.collect_env

NOTE: I have seen a few people fail to get this working because they did not run the commands one at a time. I condensed this down and tested so there are technically only 2 lines to run.
```

### Step 10: Test run PyTorch

```
From within WSL Ubuntu 22.04 do the following:

cd ~
touch test.py
nano test.py

Paste in the following code, be prepared to monitor GPU utilization to validate the
workload is actually using GPU and not CPU:

######################################################################
#The MNIST dataset is widely used for introductory machine learning projects.
#Download MNIST dataset using torchvision
#Train the model
#Evaluate the model after training
#Validates our WSL2 environment setup and PyTorch installation for GPU acceleration

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Define transforms to normalize the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 3. Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

# 4. Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 1 input channel (grayscale), 32 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 window
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer (flattened after 2 pooling operations)
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes (digits 0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Train the model
for epoch in range(5):  # 5 epochs
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100}')
            running_loss = 0.0

print('Finished Training')

# 7. Evaluate the model on the test set
correct = 0
total = 0
model.eval()  # Set model to evaluation mode (disables dropout, etc.)
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
######################################################################

Save and exit the file: Ctrl+O, Enter, Ctrl+X

Run the following command:

python3 test.py

You should see the loss function and accuracy print out as the model trains and evaluates. If you see this then you have successfully setup WSL2 with PyTorch for GPU acceleration.

You should also validate the GPU is being loaded via Adrenalin, HWMon or HWInfo64 etc…
```

### Step 11: Stable Diffusion WebUI

```
Repo we will be using: https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu

If for some reason this version is not available I have a copy here: https://drive.google.com/file/d/1Ila5AQjgBmr2g6b1CeC2P_m5vItj8FvP/view?usp=sharing

We are mostly already setup since we have ROCm installed and our environment vars loaded.

Create a folder where we can download the stable diffusion repo. I will be using F:\AI\tiger

NOTE: conda is a type of virtual environment that lets you run different versions of python and packages. We will be using this to run the webui.

Open Miniconda3 and run the following:

f:
cd ai
cd tiger
conda create --name tiger python=3.10.6 -y
conda activate tiger
git clone https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu
cd stable-diffusion-webui-amdgpu

Edit the webui-user.bat file and update this line to show the following:

set COMMANDLINE_ARGS=--use-zluda

Save and exit the file

Run the following command from the stable-diffusion-webui-amdgpu folder:

webui-user.bat

This will take a while to run the first time.
```

### Step 12: BONUS – Install Jupyter Labs & Link to Windows Github Desktop

```
To keep my machine learning and class taking more in-line with my drive to run all the things on Radeon, I decided to move from Google Colab to just self-hosted Jupyter Lab Notebooks. Now I will not be constrained to unavailable GPUs and obviously everything runs at a much greater speed.

From your WSL Ubuntu 22.04 instance run the following:

pip install jupyterlab

One thing you will want to do is probably run git or my preference Github desktop in windows. My github folder in windows is located here:

C:\Users\user\Documents\GitHub\public

We will want to create a symbolic link from your Ubuntu home folder: cd ~

Create a symbolic link from WSL2 Ubuntu to the windows mount for your specific github repo:

ln -s /mnt/c/Users/user/Documents/GitHub/public/ ~/github

Update your PATH:

export PATH="$HOME/.local/bin:$PATH"

Start Jupyter Lab:

jupyter lab

Navigate to: JupyterLab

Keep in mind you will need to install packages via PIP or however you normally install packages, things like matplotlib, sci-kit etc… pytorch/matplotlib is already installed from several steps above.

pip install scikit-learn
pip install pandas
```