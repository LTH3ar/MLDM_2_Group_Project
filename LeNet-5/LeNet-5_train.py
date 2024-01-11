# %% md
# 1. Importing Libraries
# %%
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

# %% md
# 2. Setting Device
# %%
# Setting Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)
# %% md
# 3. Preparing Input Data
# %%
# Preparing Input Data
# prepare the dataset MNIST(1x28x28) for LeNet
param_transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # Normalize to [-1, 1] range
])
# %%
# Download the dataset
train_val_dataset = datasets.MNIST(root='./dataset', train=True, transform=param_transform, download=True)

# Dataset summary
print('train_val_dataset length:', len(train_val_dataset))
print('train_val_dataset shape:', train_val_dataset[0][0].shape)
# %%
# Split the dataset into train and validation
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

# Dataset summary
print('train_dataset length:', len(train_dataset))
print('val_dataset length:', len(val_dataset))
# %%
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


# %% md
# 4. Defining Model
# %%
# Defining Model
# LeNet-5 architecture implementation
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.feature = nn.Sequential(
            # Convolutional layers

            # ============================================================================== #
            # First conv layer
            # input: 1 x 28 x 28 --> padding = 2 --> 1 x 32 x 32 --> 6 x 28 x 28
            # nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            # activation function
            nn.Tanh(),
            # pooling layer 14 x 14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ============================================================================== #

            # ============================================================================== #
            # Second conv layer
            # input: 6 x 14 x 14 --> 16 x 10 x 10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # activation function
            nn.Tanh(),
            # pooling layer 5 x 5
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ============================================================================== #
        )

        # Classifier
        self.classifier = nn.Sequential(
            # Fully connected layers

            # ============================================================================== #
            # First fc layer
            # input: 16 x 5 x 5 = 400 --> 120
            # flatten
            nn.Flatten(),
            # fc layer
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            # activation function
            nn.Sigmoid(),  # sigmoid
            # ============================================================================== #

            # ============================================================================== #
            # Second fc layer
            nn.Linear(in_features=120, out_features=84),

            # activation function
            nn.Sigmoid(),  # sigmoid
            # ============================================================================== #

            # ============================================================================== #
            # Third fc layer
            nn.Linear(in_features=84, out_features=10),
            # ============================================================================== #
            nn.Softmax(dim=1)
        )

    # Forward function
    def forward(self, x):
        return self.classifier(self.feature(x))


# %%
# Create model
model = LeNet5().to(device)
print(model)
# %%
# Model summary
# Detailed layer-wise summary
# summary(model, input_size=(1, 1, 28, 28), verbose=2, device=device)
summary(model, input_size=(1, 1, 32, 32), verbose=2, device=device)
# %%
# Optimizer and loss function
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss().to(device)
accuracy = Accuracy(task='multiclass', num_classes=10).to(device)
# %% md
# 5. Training
# %%
# Training
# Log training process to TensorBoard
date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
log_dir = os.path.join('train_logs', date_time)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
# %%
# Training Parameters
NUM_EPOCHS = 40
NUM_BATCHES = len(train_loader)
NUM_BATCHES_VAL = len(val_loader)
# %%
# Save the model checkpoint
if not os.path.exists('models'):
    os.mkdir('models')
VERSION = 1
# %%
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
        print(
            f'Train Epoch: {epoch_count} [{batch_idx}/{NUM_BATCHES} ({100. * batch_idx / NUM_BATCHES:.0f}%)]\tLoss: {loss.item():.6f}')

    # Calculate average loss and accuracy over an epoch
    avg_loss_train = loss_function(torch.cat(y_pred_train), torch.cat(y_true_train)).item()
    avg_acc_train = accuracy(torch.cat(y_pred_train), torch.cat(y_true_train)) * 100

    print(
        f'Epoch: {epoch_count}\tAverage Train Loss: {avg_loss_train:.6f}\tAverage Train Accuracy: {avg_acc_train:.6f}')

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
            print(
                f'Validation Epoch: {epoch_count} [{batch_idx}/{NUM_BATCHES_VAL} ({100. * batch_idx / NUM_BATCHES_VAL:.0f}%)]\tLoss: {loss.item():.6f}')

    # Calculate average loss and accuracy over an epoch
    avg_loss_val = loss_function(torch.cat(y_pred_val), torch.cat(y_true_val)).item()
    avg_acc_val = accuracy(torch.cat(y_pred_val), torch.cat(y_true_val)) * 100

    print(
        f'Epoch: {epoch_count}\tAverage Validation Loss: {avg_loss_val:.6f}\tAverage Validation Accuracy: {avg_acc_val:.6f}')

    # Log average loss and accuracy of an epoch
    writer.add_scalars(main_tag='Accuracy train/validation',
                       tag_scalar_dict={'TRAIN': avg_acc_train, 'VAL': avg_acc_val}, global_step=epoch_count)
    writer.add_scalars(main_tag='Loss train/validation', tag_scalar_dict={'TRAIN': avg_loss_train, 'VAL': avg_loss_val},
                       global_step=epoch_count)

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
# %%
# Close the writer
writer.flush()
writer.close()
# %%
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