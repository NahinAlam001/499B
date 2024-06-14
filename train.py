import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data_loading import load_data
from imports import my_skin_model, device

# Custom dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return {'image': image, 'mask': mask}

# Load data
image_dir = "drive/MyDrive/Dataset/images"
mask_dir = "drive/MyDrive/Dataset/masks"
images, masks = load_data(image_dir, mask_dir)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = SkinLesionDataset(images, masks, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the model, loss function, and optimizer
model = my_skin_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs = data['image'].to(device)
            labels = data['mask'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

    print('Finished Training')

# Train the model
train_model(model, train_loader, criterion, optimizer, device)
