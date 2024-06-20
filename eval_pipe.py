import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
data_dir = 'Dataset'
csv_file = 'data.csv'
checkpoint_path = 'densenet_checkpoint_4.pth'

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Map class names to indices
        self.class_names = self.data_frame['dx'].unique()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.data_frame.iloc[idx, 0] + '.bmp')
        image = Image.open(img_name).convert('RGBA')
        label = self.data_frame.iloc[idx, 1]
        label = self.class_to_idx[label]  # Convert label to index

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Custom transform to convert RGBA to RGB and then to tensor with 4 channels
class RGBAToTensor:
    def __call__(self, img):
        # Convert RGBA to RGB
        rgb_image = img.convert('RGB')
        
        # Convert RGB to tensor with 3 channels
        rgb_tensor = transforms.ToTensor()(rgb_image)
        
        # Create an alpha channel (all ones)
        alpha_tensor = torch.ones_like(rgb_tensor[0:1, :, :])
        
        # Concatenate RGB and alpha channel
        rgba_tensor = torch.cat([rgb_tensor, alpha_tensor], dim=0)
        
        return rgba_tensor

# Define transformations (adapted for RGBA images)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    RGBAToTensor(),
])

# Load the dataset
full_dataset = CustomDataset(csv_file=csv_file, root_dir=data_dir, transform=transform)

# Split the dataset into training, validation, and testing sets
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
class_names = full_dataset.class_names

# Load pre-trained DenseNet model
model = models.densenet121(pretrained=False)  # Ensure to set pretrained to False
conv0_weight = model.features.conv0.weight.data
new_conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Initialize weights of the new conv0 layer
with torch.no_grad():
    new_conv0.weight[:, :3, :, :].copy_(conv0_weight)
    new_conv0.weight[:, 3:, :, :] = torch.randn_like(new_conv0.weight[:, 3:, :, :])

# Replace old conv0 with new one
model.features.conv0 = new_conv0

# Update model's classifier
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(class_names))

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load saved checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)

# Evaluation function
def evaluate_model(model, dataloader, dataset_size):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / dataset_size
    return accuracy, all_preds, all_labels

# Evaluate model on testing set
test_accuracy, test_preds, test_labels = evaluate_model(model, dataloaders['test'], dataset_sizes['test'])
print(f'Testing Accuracy: {test_accuracy:.4f}')

# Calculate Precision, Recall, F1 score
precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
print(f'Testing Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Testing")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
print(classification_report(test_labels, test_preds, target_names=class_names))
