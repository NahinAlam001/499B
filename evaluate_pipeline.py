import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define paths
images_dir = "/content/Dataset/images"  
masks_dir = "/content/Dataset/masks"   
csv_file = "data.csv"                  
checkpoint_file = "classification.pth"  

# Hyperparameters and constants
batch_size = 4
target_size = (256, 256)

# Data transformations (similar to training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Custom dataset class (similar to training)
class CustomImageItemDataset(Dataset):
    def __init__(self, images_dir, masks_dir, csv_file, transform=None, target_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = target_size
        unique_labels = self.data['dx'].unique()
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, f"{self.data.iloc[idx, 0]}.bmp")
        mask_name = os.path.join(self.masks_dir, f"{self.data.iloc[idx, 0]}_lesion.bmp")
        if not os.path.isfile(img_name) or not os.path.isfile(mask_name):
            raise ValueError(f"Image or mask file not found for index {idx}")
        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Error loading image at {img_name}")
        if mask is None:
            raise ValueError(f"Error loading mask at {mask_name}")

        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        label_str = self.data.iloc[idx, 1]
        if label_str not in self.label_map:
            raise ValueError(f"Unknown label: {label_str}")

        label = self.label_map[label_str]
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return {"image": image, "mask": mask, "label": label, "mask_name": mask_name}

# Model definition (same as training)
class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CustomImageItemDataset(images_dir, masks_dir, csv_file, transform=transform)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)

    # Load trained model
    model = DenseNetClassifier(num_classes=len(dataset.label_map))
    model = model.to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['classifier_state_dict'])

    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
