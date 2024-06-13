import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir
from model import UNet

H, W = 256, 256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.transpose(x, (2, 0, 1))  ## Convert to (3, 256, 256)
    x = np.expand_dims(x, axis=0)   ## Add batch dimension
    x = torch.tensor(x, dtype=torch.float32)
    return ori_x, x  ## (1, 3, 256, 256)

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.int32)  ## (256, 256)
    x = torch.tensor(x, dtype=torch.int64)
    return ori_x, x

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred * 255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    torch.manual_seed(42)

    """ Folder for saving results """
    create_dir("results")

    """ Load the model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load("files/model.pth", map_location=device))
    model.eval()

    """ Load the test data """
    dataset_path = "./isic-challenge-2018/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    SCORE = []
    with torch.no_grad():
        for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
            """ Extracting the image name """
            name = os.path.basename(x)

            """ Read the image and mask """
            ori_x, x = read_image(x)
            ori_y, y = read_mask(y)

            """ Predicting the mask """
            x = x.to(device)
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
            y_pred = y_pred.cpu().numpy()[0, 0]  ## Remove batch and channel dimensions
            y_pred = y_pred.astype(np.int32)

            """ Saving the predicted mask """
            save_image_path = f"results/{name}"
            save_results(ori_x, ori_y, y_pred, save_image_path)

            """ Flatten the array """
            y = y.numpy().flatten()
            y_pred = y_pred.flatten()

            """ Calculating metrics values """
            acc_value = accuracy_score(y, y_pred)
            f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
            jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
            recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
            precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
            SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ mean metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")
