import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor, SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
from data_loader import load_data

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
        prompt = get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs

def train_model(dataset, model_path="skin_model_ISIC_2017_checkpoint.pth", num_epochs=1, batch_size=2, learning_rate=1e-5):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    assert isinstance(model, SamModel), "Loaded model is not of type SamModel"
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

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

if __name__ == "__main__":
    image_dir = "/content/isic-challenge-2017/ISIC2017_Task1-2_Training_Input"
    mask_dir = "/content/isic-challenge-2017/ISIC2017_Task1-2_Training_GroundTruth"
    dataset = load_data(image_dir, mask_dir)
    train_model(dataset)
