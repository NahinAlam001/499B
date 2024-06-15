import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor, SamModel
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, recall_score, confusion_matrix
from data_loader import load_data
from fvcore.nn import FlopCountAnalysis

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

def evaluate_model(model_path="/content/drive/MyDrive/skin_model_ISIC_2017_checkpoint.pth", batch_size=2):
    image_dir = "/content/isic-challenge-2017/ISIC2017_Task1-2_Training_Input"
    mask_dir = "/content/isic-challenge-2017/ISIC2017_Task1-2_Training_GroundTruth"
    dataset = load_data(image_dir, mask_dir)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    eval_dataset = SAMDataset(dataset=dataset, processor=processor)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    jaccard_scores = []
    dice_scores = []
    accuracies = []
    recalls = []
    f1_scores = []
    total_flops = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1).cpu().numpy()
            ground_truth_masks = batch["ground_truth_mask"].numpy()

            for pred_mask, true_mask in zip(predicted_masks, ground_truth_masks):
                pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
                true_mask_bin = (true_mask > 0.5).astype(np.uint8)

                # Jaccard Score
                jaccard = jaccard_score(true_mask_bin.flatten(), pred_mask_bin.flatten())
                jaccard_scores.append(jaccard)

                # Dice Score
                intersection = np.logical_and(true_mask_bin, pred_mask_bin).sum()
                dice = 2. * intersection / (true_mask_bin.sum() + pred_mask_bin.sum())
                dice_scores.append(dice)

                # Accuracy
                accuracy = accuracy_score(true_mask_bin.flatten(), pred_mask_bin.flatten())
                accuracies.append(accuracy)

                # Recall
                recall = recall_score(true_mask_bin.flatten(), pred_mask_bin.flatten())
                recalls.append(recall)

                # F1 Score
                f1 = f1_score(true_mask_bin.flatten(), pred_mask_bin.flatten())
                f1_scores.append(f1)

            # Calculate FLOPs
            flops = FlopCountAnalysis(model, batch["pixel_values"].to(device)).total()
            total_flops += flops

    mean_jaccard = np.mean(jaccard_scores)
    mean_dice = np.mean(dice_scores)
    mean_accuracy = np.mean(accuracies)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    mean_flops = total_flops / len(eval_dataloader)

    print(f'Mean Jaccard Score: {mean_jaccard}')
    print(f'Mean Dice Score: {mean_dice}')
    print(f'Mean Accuracy: {mean_accuracy}')
    print(f'Mean Recall: {mean_recall}')
    print(f'Mean F1 Score: {mean_f1}')
    print(f'Mean FLOPs: {mean_flops}')

if __name__ == "__main__":
    evaluate_model()
