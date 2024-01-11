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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
#%% md
# 2. Setting up the device
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
# 3. Preparing the test_data
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
# Load the dataset
test_data = datasets.MNIST(
    root='./dataset', 
    train=False, 
    transform=param_transform, 
    download=True)

test_data_info = datasets.MNIST(
    root='./dataset', 
    train=True,
    download=False)
print("MNIST dataset information:")
print(" - Number of training samples: {}".format(len(test_data_info)))
print(" - Number of test samples: {}".format(len(test_data)))
print(" - Image Shape: {}".format(test_data[0][0].shape))
print(" - Number of classes: {}".format(len(test_data.classes)))
print(" - Samples number all classes: {}".format(test_data.targets.bincount()))
# info about the dataset
print(test_data_info)

# plot the train dataset samples count for each class
fig = plt.figure(figsize=(10, 10))
plt.bar(test_data_info.classes, test_data_info.targets.bincount())
plt.title('Train dataset samples count for each class')
plt.xlabel('Classes')
plt.ylabel('Samples count')
plt.savefig('train_dataset_samples_count.png')
plt.show()
plt.close()

# Dataset summary
print('Test dataset:')
print(' - Number of datapoints: {}'.format(len(test_data)))
print(' - Image Shape: {}'.format(test_data[0][0].shape))
print(' - Number of classes: {}'.format(len(test_data.classes)))
print(" - Samples number all classes: {}".format(test_data.targets.bincount()))

# plot the test dataset samples count for each class
fig = plt.figure(figsize=(10, 10))
plt.bar(test_data.classes, test_data.targets.bincount())
plt.title('Test dataset samples count for each class')
plt.xlabel('Classes')
plt.ylabel('Samples count')
plt.savefig('test_dataset_samples_count.png')
plt.show()
plt.close()
#%%
# Dataset for visualization
# get the 10 images from the test set for visualization
vis_size = 10
test_data_vis = torch.utils.data.Subset(test_data, range(vis_size))

# Dataset summary
print('Test dataset for visualization:')
print(' - Number of datapoints: {}'.format(len(test_data_vis)))
print(' - Image Shape: {}'.format(test_data_vis[0][0].shape))

# loader
test_loader_vis = DataLoader(test_data_vis, batch_size=vis_size, shuffle=True)
#%%
# Create dataloaders
if torch.cuda.is_available():
    BATCH_SIZE = 128
elif torch.backends.mps.is_available():
    BATCH_SIZE = 128
else:
    BATCH_SIZE = 64
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Dataloader summary
print('Test dataloader:')
print(' - Number of batches: {}'.format(len(test_loader)))
#%% md
# 4. Loading the model
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
# Define the loss function
loss_fn = nn.CrossEntropyLoss().to(device)
accuracy = Accuracy(task='multiclass', num_classes=10).to(device)
#%%
# Log the test
date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
#%%
models_lst = os.listdir('models')
models_lst.sort()
print(models_lst)
# create the log directory
test_folder_name = f"test_{date_time}"
log_dir = os.path.join('test_logs', test_folder_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
#%%
# Test the model
y_true = []
y_pred = []

loss_lst = []
acc_lst = []
for model_name in models_lst:
    # Load the model
    model = AlexNet().to(device)
    # Load the model weights
    model.load_state_dict(torch.load(os.path.join('models', model_name), map_location=device))
    # Model summary
    # summary(model, input_size=(1, 1, 28, 28), verbose=2, device=device)
    summary(model, input_size=(1, 3, 227, 227), verbose=2, device=device)
    # Calculate model weights size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model size: {model_size} parameters')
    print('Model weights size: {:.2f} MB'.format(sum(p.numel() for p in model.parameters()) / (1024 * 1024)))
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        loss_tmp = []
        y_true_tmp = []
        y_pred_tmp = []
        y_true_cfm = []
        y_pred_cfm = []
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move the data to device
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = model(data)

            # save the true and predicted labels for confusion matrix
            y_true_tmp.append(target)
            y_pred_tmp.append(output)

            # save the true and predicted labels for confusion matrix
            y_true_cfm.extend(target.cpu().numpy())
            y_pred_cfm.extend(torch.argmax(output, dim=1).cpu().numpy())

        # save the true and predicted labels for confusion matrix, convert to numpy array
        y_true.append(y_true_cfm)
        y_pred.append(y_pred_cfm)

        # Calculate the average accuracy
        avg_acc = accuracy(torch.cat(y_pred_tmp), torch.cat(y_true_tmp)).item()*100
        avg_loss = loss_fn(torch.cat(y_pred_tmp), torch.cat(y_true_tmp)).item()
    print('Test: Average accuracy: {:.2f}%'.format(avg_acc))
    print('Test: Average loss: {:.2f}'.format(avg_loss))
    # for each model add the accuracy and loss to the tensorboard so that we can compare them like a line
    acc_lst.append(avg_acc)
    loss_lst.append(avg_loss)

    # clean the GPU memory
    # clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    else:
        torch.empty_cache()
    # release the model from GPU memory
    model = None

for i in range(len(models_lst)):
    writer.add_scalar(tag="Accuracy", scalar_value=acc_lst[i], global_step=int((i + 1) * 10))
    writer.add_scalar(tag="Loss", scalar_value=loss_lst[i], global_step=int((i + 1) * 10))
#%%
# Confusion matrix all test data
for i in range(len(models_lst)):
    # Confusion matrix
    cm = confusion_matrix(y_true[i], y_pred[i])
    print(cm)
    # Classification report
    print(classification_report(y_true[i], y_pred[i], target_names=test_data.classes))
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(log_dir, f'confusion_matrix_{models_lst[i]}.png'))
    plt.show()
    with open(os.path.join(log_dir, f'classification_report_{models_lst[i]}.txt'), 'w') as f:
        f.write(classification_report(y_true[i], y_pred[i], target_names=test_data.classes))
        f.write('\n')
        f.write('Confusion matrix:\n')
        f.write(str(cm))
#%%
# Visualize the bar chart accuracy of models, V1, V2, V3, V4 (range 97-100%) add acc value to the top of the bar
labels = []
for i in range(len(models_lst)):
    labels.append(f'V{i + 1}')
print(labels)
fig, ax = plt.subplots()
ax.bar(labels, acc_lst)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Models')
ax.set_title('Accuracy of models')
ax.set_ylim([97, 100])
for i, v in enumerate(acc_lst):
    ax.text(i - 0.1, v + 0.1, str(round(v, 2)), color='black', fontweight='bold')
plt.savefig(os.path.join(log_dir, 'accuracy_of_models.png'))
plt.show()
#%%
# Close the writer
writer.flush()
writer.close()
#%%
# Release GPU memory cache
# clear cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
else:
    torch.empty_cache()

# Release model from GPU memory
model = None

# release all loaded data
data_vis = None
target_vis = None
data = None
target = None
data_loader = None
data_loader_vis = None
test_data = None
test_data_vis = None
test_loader = None
test_loader_vis = None

print('Done!')
