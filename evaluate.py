import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor, SamModel
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, recall_score
from data_loader import load_data
from tqdm import tqdm

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

def evaluate_model(model_path="skin_model_PH2_SAM_checkpoint.pth", batch_size=2):
    image_dir = "./Dataset/images"
    mask_dir = "./Dataset/masks"
    dataset = load_data(image_dir, mask_dir)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    eval_dataset = SAMDataset(dataset=dataset, processor=processor)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    jaccard_scores = []
    dice_scores = []
    accuracies = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            outputs = model(pixel_values=pixel_values,
                            input_boxes=input_boxes,
                            multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            predicted_masks = (predicted_masks > 0.5).float()

            for pred_mask, true_mask in zip(predicted_masks, ground_truth_masks):
                pred_mask = pred_mask.cpu().numpy().flatten()
                true_mask = true_mask.cpu().numpy().flatten()

                jaccard = jaccard_score(true_mask, pred_mask)
                jaccard_scores.append(jaccard)

                intersection = np.logical_and(true_mask, pred_mask).sum()
                dice = 2. * intersection / (true_mask.sum() + pred_mask.sum())
                dice_scores.append(dice)

                accuracy = accuracy_score(true_mask, pred_mask)
                accuracies.append(accuracy)

                recall = recall_score(true_mask, pred_mask)
                recalls.append(recall)

                f1 = f1_score(true_mask, pred_mask)
                f1_scores.append(f1)

    print(f'Mean Jaccard Score: {np.mean(jaccard_scores)}')
    print(f'Mean Dice Score: {np.mean(dice_scores)}')
    print(f'Mean Accuracy: {np.mean(accuracies)}')
    print(f'Mean Recall: {np.mean(recalls)}')
    print(f'Mean F1 Score: {np.mean(f1_scores)}')

if __name__ == "__main__":
    evaluate_model()
