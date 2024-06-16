import os
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch import nn
from PIL import Image
from transformers import SamProcessor, SamModel
import numpy as np
from tqdm import tqdm
from pathlib import Path

class SAMClassificationDataset(Dataset):
    def __init__(self, image_paths, processor, transform=None):
        self.image_paths = image_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_transformed = self.transform(image)

        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "image_path": image_path,
            "image": image_transformed,
            "pixel_values": pixel_values
        }

def load_image_paths(images_dir):
    image_paths = []
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith(('.bmp', '.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_segmentations_and_combine(dataset, sam_model, device):
    combined_images = []
    image_paths = []

    for item in tqdm(dataset):
        pixel_values = item["pixel_values"].to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = sam_model(pixel_values=pixel_values)
        seg_mask = outputs.pred_masks.squeeze().cpu().numpy()

        image = item["image"].numpy()
        combined_image = np.concatenate((image, seg_mask[np.newaxis, ...]), axis=0)
        combined_images.append(combined_image)
        image_paths.append(item["image_path"])

    return torch.tensor(combined_images), image_paths

def classify_images(image_paths, sam_model_path="sam_model_checkpoint.pth", densenet_path="densenet_checkpoint.pth", output_dir="./ImageFolder", batch_size=2):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SAMClassificationDataset(image_paths=image_paths, processor=processor, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    sam_model.load_state_dict(torch.load(sam_model_path))
    sam_model.to(device)
    sam_model.eval()

    combined_images, image_paths = get_segmentations_and_combine(dataset, sam_model, device)

    densenet = models.densenet121(pretrained=False)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, len(set(Path(img).parent.stem for img in image_paths)))  # Adjusting for the number of classes
    densenet.load_state_dict(torch.load(densenet_path))
    densenet.to(device)
    densenet.eval()

    combined_images = combined_images.to(device)

    predictions = []
    with torch.no_grad():
        outputs = densenet(combined_images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

    class_names = sorted(list(set(Path(img).parent.stem for img in image_paths)))
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    for img_path, pred in zip(image_paths, predictions):
        class_name = class_names[pred]
        shutil.copy(img_path, os.path.join(output_dir, class_name))

    print("Images have been classified and copied to the respective class directories.")

if __name__ == "__main__":
    image_dir = "./Dataset/images"
    sam_model_path = "path/to/sam_model_checkpoint.pth"
    densenet_path = "path/to/densenet_checkpoint.pth"
    output_dir = "./ImageFolder"
