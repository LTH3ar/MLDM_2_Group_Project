#%% md
# 1. Importing Libraries
#%%
# Importing Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
from torchmetrics import Accuracy
from torchinfo import summary
import numpy as np
import os
import datetime
#%% md
# 2. Setting Device
#%%
# Setting Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)
#%% md
# 3. Preparing Input Data
#%%
# Preparing Input Data
# prepare the dataset MNIST(1x28x28) -> (3x224x224) for AlexNet
# Upscale the grayscale images to RGB size
param_transform = transforms.Compose([
    transforms.Pad(2),
    transforms.Resize((227, 227)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # Normalize to [-1, 1] range
])
#%%
# Download the dataset
train_val_dataset = datasets.MNIST(root='./dataset', train=True, transform=param_transform, download=True)

# Dataset summary
print('train_val_dataset length:', len(train_val_dataset))
print('train_val_dataset shape:', train_val_dataset[0][0].shape)
#%%
# Split the dataset into train and validation
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

# Dataset summary
print('train_dataset length:', len(train_dataset))
print('val_dataset length:', len(val_dataset))
#%%
# Create dataloaders
if torch.cuda.is_available():
    BATCH_SIZE = 128
elif torch.backends.mps.is_available():
    BATCH_SIZE = 128
else:
    BATCH_SIZE = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# dataloaders summary
print('train_loader length:', len(train_loader))
print('val_loader length:', len(val_loader))
#%% md
# 4. Defining Model
#%%
# Define the model AlexNet specific for the transformed MNIST
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # ============================================================================== #
            # 1st conv layer
            # input: 3x224x224 (upscaled from 1x28x28)
            # output: 96x55x55
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0, ),
            # activation function: ReLU
            nn.ReLU(),
            # max pooling layer with kernel size 3 and stride 2
            # output: 96x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ============================================================================== #
            
            # ============================================================================== #
            # 2nd conv layer
            # input: 96x27x27
            # output: 256x27x27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            # activation function: ReLU
            nn.ReLU(),
            # max pooling layer with kernel size 3 and stride 2
            # output: 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ============================================================================== #
            
            # ============================================================================== #
            # 3rd conv layer
            # input: 256x13x13
            # output: 384x13x13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            # activation function: ReLU
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 4th conv layer
            # input: 384x13x13
            # output: 384x13x13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            # activation function: ReLU
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 5th conv layer
            # input: 384x13x13
            # output: 256x13x13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            # activation function: ReLU
            nn.ReLU(),
            # max pooling layer with kernel size 3 and stride 2
            # output: 256x6x6
            nn.MaxPool2d(kernel_size=3, stride=2)
            # ============================================================================== #
        )

        self.classifier = nn.Sequential(
            # flatten
            nn.Flatten(), # 256*5*5 = 6400
            # ============================================================================== #
            # 1st fc layer Dense: 4096 fully connected neurons
            nn.Dropout(p=0.5), # dropout layer with p=0.5
            nn.Linear(in_features=256 * 6 * 6, out_features=4096), # 256*5*5
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 2nd fc layer Dense: 4096 fully connected neurons
            nn.Dropout(p=0.5), # dropout layer with p=0.5
            nn.Linear(in_features=4096, out_features=4096), # 4096
            nn.ReLU(),
            # ============================================================================== #
            
            # ============================================================================== #
            # 3rd fc layer Dense: 10 fully connected neurons
            nn.Linear(in_features=4096, out_features=num_classes) # 4096
            # ============================================================================== #

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
#%%
# Create model
model = AlexNet().to(device)
print(model)
#%%
# Model summary
# Detailed layer-wise summary
#summary(model, input_size=(1, 3, 224, 224), verbose=2, device=device)
summary(model, input_size=(1, 3, 227, 227), verbose=2, device=device)
#%%
# Optimizer and loss function
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss().to(device)
accuracy = Accuracy(task='multiclass', num_classes=10).to(device)
#%% md
# 5. Training
#%%
# Training
# Log training process to TensorBoard
date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
log_dir = os.path.join('train_logs', date_time)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
#%%
# Training Parameters
NUM_EPOCHS = 40
NUM_BATCHES = len(train_loader)
NUM_BATCHES_VAL = len(val_loader)
#%%
# Save the model checkpoint
if not os.path.exists('models'):
    os.mkdir('models')
VERSION = 1
#%%
# Training loop
for epoch in range(NUM_EPOCHS):
    y_pred_train = []
    y_true_train = []
    y_pred_val = []
    y_true_val = []
    y_pred_test = []
    y_true_test = []
    epoch_count = epoch + 1
    # Training
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data = data.to(device)
        target = target.to(device)

        # Forward
        output = model(data)
        loss = loss_function(output, target)
        
        # add y_pred and y_true
        y_pred_train.append(output)
        y_true_train.append(target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # log training
        print(f'Train Epoch: {epoch_count} [{batch_idx}/{NUM_BATCHES} ({100. * batch_idx / NUM_BATCHES:.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate average loss and accuracy over an epoch
    avg_loss_train = loss_function(torch.cat(y_pred_train), torch.cat(y_true_train)).item()
    avg_acc_train = accuracy(torch.cat(y_pred_train), torch.cat(y_true_train))*100
    
    print(f'Epoch: {epoch_count}\tAverage Train Loss: {avg_loss_train:.6f}\tAverage Train Accuracy: {avg_acc_train:.6f}')

    # Validation
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            # Move data to device
            data = data.to(device)
            target = target.to(device)

            # Forward
            output = model(data)
            
            # add y_pred and y_true
            y_pred_val.append(output)
            y_true_val.append(target)

            # log validation
            print(f'Validation Epoch: {epoch_count} [{batch_idx}/{NUM_BATCHES_VAL} ({100. * batch_idx / NUM_BATCHES_VAL:.0f}%)]\tLoss: {loss.item():.6f}')
                
    # Calculate average loss and accuracy over an epoch
    avg_loss_val = loss_function(torch.cat(y_pred_val), torch.cat(y_true_val)).item()
    avg_acc_val = accuracy(torch.cat(y_pred_val), torch.cat(y_true_val))*100
    
    print(f'Epoch: {epoch_count}\tAverage Validation Loss: {avg_loss_val:.6f}\tAverage Validation Accuracy: {avg_acc_val:.6f}')

    # Log average loss and accuracy of an epoch 
    writer.add_scalars(main_tag='Accuracy train/validation', tag_scalar_dict={'TRAIN': avg_acc_train, 'VAL': avg_acc_val}, global_step=epoch_count)
    writer.add_scalars(main_tag='Loss train/validation', tag_scalar_dict={'TRAIN': avg_loss_train, 'VAL': avg_loss_val}, global_step=epoch_count)
    

    if epoch_count % 10 == 0:
        MODEL_NAME = f'LeNet5_v{VERSION}_{date_time}.pth'
        torch.save(model.state_dict(), os.path.join('models', MODEL_NAME))
        print(f'Saved PyTorch Model State to {MODEL_NAME}')
        VERSION += 1
        
# clear cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
else:
    torch.empty_cache()
features = None
targets = None
#%%
# Close the writer
writer.flush()
writer.close()
#%%
model = None
model_loaded = None

# release all loaders
train_loader = None
val_loader = None
test_loader = None

# release all variables
optimizer = None
loss_fn = None
accuracy = None

print('Released all variables')
