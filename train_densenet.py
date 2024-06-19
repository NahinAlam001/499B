import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Set paths
data_dir = 'Dataset'
csv_file = 'data.csv'
save_path = 'densenet_checkpoint_4.pth'

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

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.class_names

# Load pre-trained DenseNet model
model = models.densenet121(pretrained=True)

# Modify the first convolutional layer to accept 4 input channels
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

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training the model
num_epochs = 25
best_model_wts = model.state_dict()
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print()

print('Best val Acc: {:4f}'.format(best_acc))

# Load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), save_path)
print('Model saved to', save_path)
