import os
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor, SamModel

class SAMClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, processor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_transformed = self.transform(image)

        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "image": image_transformed,
            "pixel_values": pixel_values,
            "label": label
        }

def load_data(data_dir):
    image_paths = []
    labels = []
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(os.listdir(data_dir))}

    for cls_name, idx in class_to_idx.items():
        cls_dir = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.endswith(".jpg") or img_name.endswith(".bmp"):
                image_paths.append(os.path.join(cls_dir, img_name))
                labels.append(idx)

    return image_paths, labels

def get_segmentations_and_combine(dataset, sam_model, device):
    combined_images = []
    labels = []

    for item in tqdm(dataset):
        pixel_values = item["pixel_values"].to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = sam_model(pixel_values=pixel_values)
        seg_mask = outputs.pred_masks.squeeze().cpu().numpy()

        image = item["image"].numpy()
        combined_image = np.concatenate((image, seg_mask[np.newaxis, ...]), axis=0)
        combined_images.append(combined_image)
        labels.append(item["label"])

    return torch.tensor(combined_images), torch.tensor(labels)

def classify_images(image_paths, labels, sam_model_path="sam_model_checkpoint.pth", densenet_path="densenet_checkpoint.pth", batch_size=2):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SAMClassificationDataset(image_paths=image_paths, labels=labels, processor=processor, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    sam_model.load_state_dict(torch.load(sam_model_path))
    sam_model.to(device)
    sam_model.eval()

    combined_images, labels = get_segmentations_and_combine(dataset, sam_model, device)

    densenet = models.densenet121(pretrained=False)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, len(set(labels)))  # Adjusting for the number of classes
    densenet.load_state_dict(torch.load(densenet_path))
    densenet.to(device)
    densenet.eval()

    combined_images = combined_images.to(device)
    labels = labels.to(device)

    predictions = []
    with torch.no_grad():
        outputs = densenet(combined_images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

    return predictions

if __name__ == "__main__":
    data_dir = "./data/images"
    image_paths, labels = load_data(data_dir)

    sam_model_path = "path/to/sam_model_checkpoint.pth"
    densenet_path = "path/to/densenet_checkpoint.pth"

    predictions = classify_images(image_paths, labels, sam_model_path, densenet_path)
    print("Predictions:", predictions)
