import os
import csv
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import SamProcessor, SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm

# Define your classes from data.csv
class_names = ["Common Nevus", "Atypical Nevus", "Melanoma"]

class_name_to_label = {class_name: i for i, class_name in enumerate(class_names)}

def load_data(image_dir, mask_dir, csv_path='data.csv'):
    # Load class labels from CSV
    image_to_class = {}
    with open(csv_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_to_class[row['image_id']] = row['dx']

    # Assuming image and mask directories have corresponding files
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    dataset = []
    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]  # Extract image ID without extension
        image_path = os.path.join(image_dir, image_file)
        mask_file = image_file.replace(".bmp", "_lesion.bmp")  # Adjust for your naming convention
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path) and image_id in image_to_class:
            dataset.append({
                "image": image_path,
                "label": mask_path,
                "class": image_to_class[image_id]
            })
        else:
            print(f"Mask not found for image: {image_file}")

    return dataset

# Function to get bounding box for SAM
def get_bounding_box(ground_truth_map):
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

# Function to extract features using DenseNet
def extract_features_densenet(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    densenet = models.densenet121(pretrained=True)
    densenet.eval()
    with torch.no_grad():
        features = densenet.features(transform(image).unsqueeze(0))
    return features

# Custom Dataset class to integrate SAM and DenseNet features
class SAMDenseNetDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item["image"]).convert("RGB")
        ground_truth_mask = np.array(Image.open(item["label"]).convert("L"))
        prompt = get_bounding_box(ground_truth_mask)
        
        # SAM segmentation
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask, dtype=torch.float32)
        
        # DenseNet feature extraction
        features = extract_features_densenet(image)
        features = features.squeeze(0)  # Remove batch dimension
        inputs["densenet_features"] = features
        
        # Prepare labels for classification (assuming binary classification)
        labels = torch.tensor(class_name_to_label[item["class"]], dtype=torch.long)
        
        return inputs, labels

# Function to train the model with SAM and DenseNet integration
def train_model(dataset, model_path="skin_model_PH2_SAM_DenseNet_checkpoint.pth", num_epochs=1, batch_size=2, learning_rate=1e-5):
    # SAM setup
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDenseNetDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    assert isinstance(model, SamModel), "Loaded model is not of type SamModel"
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Optimizer and loss function setup
    optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # Count unique classes
    unique_classes = set()
    for data in dataset:
        unique_classes.add(data["class"])
    
    print(f"Number of unique classes: {len(unique_classes)}")
    print(f"Unique classes: {unique_classes}")

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # SAM segmentation
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            
            # DenseNet features
            densenet_features = batch["densenet_features"].to(device)
            
            # Prepare input for CNN
            cnn_input = densenet_features.permute(0, 2, 3, 1)  # Adjust dimensions for Conv2D
            cnn_input = cnn_input.reshape(-1, 1024, 7, 7)  # Reshape to match the expected input for SimpleCNN
            
            # Classification using a simple CNN
            cnn_model = SimpleCNN()
            cnn_model.to(device)
            cnn_model.train()
            cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
            cnn_criterion = torch.nn.CrossEntropyLoss()
            
            cnn_optimizer.zero_grad()
            cnn_output = cnn_model(cnn_input)
            cnn_loss = cnn_criterion(cnn_output, batch["labels"].to(device))
            cnn_loss.backward()
            cnn_optimizer.step()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

    torch.save(model.state_dict(), model_path)

# Define a simple CNN model for classification
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)  # Adjusted input channels
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(256 * 3 * 3, 512)  # Adjusted input size for the fully connected layer
        self.fc2 = torch.nn.Linear(512, len(class_names))  # Output should match number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    image_dir = "./Dataset/images"
    mask_dir = "./Dataset/masks"
    csv_path = "data.csv"  # Path to your data.csv file
    dataset = load_data(image_dir, mask_dir, csv_path)  # Provide csv_path argument
    train_model(dataset)
