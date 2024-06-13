import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import UNet

H, W = 256, 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ISICDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (W, H)) / 255.0
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H)) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def load_data(dataset_path, split=0.2):
    images = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Training_Input", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Training_GroundTruth", "*.png")))

    train_x, test_x = train_test_split(images, test_size=split, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=split, random_state=42)
    train_x, valid_x = train_test_split(train_x, test_size=split, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    create_dir("files")

    batch_size = 4
    lr = 1e-4
    num_epochs = 5
    model_path = "files/model.pth"

    dataset_path = "./isic-challenge-2018/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    transform = transforms.ToTensor()

    train_dataset = ISICDataset(train_x, train_y, transform=transform)
    valid_dataset = ISICDataset(valid_x, valid_y, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Valid Loss: {valid_loss/len(valid_loader)}")

        torch.save(model.state_dict(), model_path)
