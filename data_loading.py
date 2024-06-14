# data_loading.py
from PIL import Image
import numpy as np
import os

def load_data(image_dir, mask_dir, image_size=(256, 256)):
    images = []
    masks = []
    for file in sorted(os.listdir(image_dir)):
        img = Image.open(os.path.join(image_dir, file))
        img = img.resize(image_size)
        images.append(np.array(img))

    for file in sorted(os.listdir(mask_dir)):
        mask = Image.open(os.path.join(mask_dir, file))
        mask = mask.resize(image_size)
        masks.append(np.array(mask))

    return np.array(images), np.array(masks)

if __name__ == "__main__":
    image_dir = "drive/MyDrive/Dataset/images"
    mask_dir = "drive/MyDrive/Dataset/masks"

    images, masks = load_data(image_dir, mask_dir)

    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)
