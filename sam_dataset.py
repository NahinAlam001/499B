# sam_dataset.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from transformers import SamProcessor
from PIL import Image
from bounding_box import get_bounding_box

class SAMDenseNetDataset(Dataset):
    def __init__(self, dataset, processor, image_size=(256, 256)):
        self.dataset = dataset
        self.processor = processor
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item["image"]).convert("RGB")
        ground_truth_mask = Image.open(item["label"]).convert("L")

        # Resize image and mask to 256x256
        image = self.transform(image)
        ground_truth_mask = self.transform(ground_truth_mask)

        assert image.shape[1:] == ground_truth_mask.shape[1:], "Image and mask sizes do not match"

        # Convert mask back to numpy array for bounding box calculation
        ground_truth_mask_np = ground_truth_mask.squeeze(0).numpy()
        prompt = get_bounding_box(ground_truth_mask_np)

        # SAM segmentation
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask_np, dtype=torch.float32)

        # DenseNet feature extraction
        features = extract_features_densenet(image)
        features = features.squeeze(0)  # Remove batch dimension
        inputs["densenet_features"] = features

        # Prepare labels for classification
        labels = torch.tensor(class_name_to_label[item["class"]], dtype=torch.long)

        return inputs, labels

def extract_features_densenet(image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Only convert to tensor
    ])
    densenet = models.densenet121(pretrained=True)
    densenet.eval()
    with torch.no_grad():
        features = densenet.features(transform(image).unsqueeze(0))
    return features
