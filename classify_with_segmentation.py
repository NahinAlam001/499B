import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from transformers import SamProcessor, SamModel
from monai.losses import DiceCELoss
from tqdm import tqdm
from PIL import Image
from statistics import mean
from sklearn.model_selection import train_test_split

class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt = self.get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs

    def get_bounding_box(self, ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
        return bbox

def train_sam_model(dataset, model_path="skin_model_PH2_SAM_checkpoint.pth", num_epochs=1, batch_size=2, learning_rate=1e-5):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0)
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

    torch.save(model.state_dict(), model_path)
    print('SAM Model saved to', model_path)

class CombinedDataset(Dataset):
    def __init__(self, image_folder, sam_model, processor, transform):
        self.image_folder = image_folder
        self.sam_model = sam_model
        self.processor = processor
        self.transform = transform
        self.classes = image_folder.classes
        self.samples = image_folder.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.sam_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=False)
        mask = outputs.pred_masks[0].cpu().numpy()
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)
        mask_tensor = self.transform(mask_image)

        combined = torch.cat([image_tensor, mask_tensor], dim=0)
        return combined, label

def train_densenet_model(image_folder, sam_model, processor, densenet_save_path="densenet_checkpoint.pth", num_epochs=25, batch_size=4, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    combined_dataset = CombinedDataset(image_folder, sam_model, processor, transform)

    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = combined_dataset.classes

    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

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

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), densenet_save_path)
    print('DenseNet Model saved to', densenet_save_path)

if __name__ == "__main__":
    image_dir = './ImageFolder'
    mask_dir = './Dataset/masks'
    sam_model_path = "skin_model_PH2_SAM_checkpoint.pth"
    densenet_path = "densenet_checkpoint.pth"

    # Load your dataset and train the SAM model
    dataset = load_data(image_dir, mask_dir)
    train_sam_model(dataset, model_path=sam_model_path)

    # Load the trained SAM model
    sam_model = SamModel.from_pretrained("Facebook/sam-vit-base")
    sam_model.load_state_dict(torch.load(sam_model_path))
    sam_model.eval()
    sam_model.to("cuda" if torch.cuda.is_available() else "cpu")
    processor = SamProcessor.from_pretrained("Facebook/sam-vit-base")

    # Load the dataset for classification
    image_folder = datasets.ImageFolder(image_dir, loader=lambda x: Image.open(x).convert("RGB"))

    # Train DenseNet model
    train_multimodal_densenet_model(image_folder, sam_model, processor, densenet_save_path=densenet_path)
