import torch
import numpy as np
from sklearn.metrics import jaccard_score, confusion_matrix

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def evaluate(model, dataset, device):
    model.eval()
    jaccard_scores = []
    dice_scores = []
    total_inference_time = 0

    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            ground_truth_mask = data['mask'].to(device)

            start_time = time.time()
            output = model(image)
            end_time = time.time()

            total_inference_time += end_time - start_time

            predicted_mask = torch.argmax(output, dim=1).cpu().numpy()
            ground_truth_mask = ground_truth_mask.cpu().numpy()

            for i in range(len(predicted_mask)):
                predicted_mask_flat = predicted_mask[i].flatten()
                ground_truth_mask_flat = ground_truth_mask[i].flatten()

                tp, fp, fn, tn = confusion_matrix(ground_truth_mask_flat, predicted_mask_flat).ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f1_score = 2 * (precision * recall) / (precision + recall)

                jaccard_scores.append(jaccard_score(ground_truth_mask_flat, predicted_mask_flat))
                dice_scores.append(dice_coefficient(ground_truth_mask_flat, predicted_mask_flat))

    mean_jaccard_score = np.mean(jaccard_scores)
    mean_dice_score = np.mean(dice_scores)
    mean_accuracy = np.mean([accuracy])
    mean_recall = np.mean([recall])
    mean_f1_score = np.mean([f1_score])

    total_inferences = len(dataset)

    print(f"Mean Jaccard Score: {mean_jaccard_score:.4f}")
    print(f"Mean Dice Score: {mean_dice_score:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1 Score: {mean_f1_score:.4f}")
    print(f"Total Inferences: {total_inferences}")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")

# Assuming 'my_skin_model' is your model and 'dataset' is the dataset
# from imports import my_skin_model, dataset, device
# evaluate(my_skin_model, dataset, device)
