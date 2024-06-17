import os
import numpy as np
from PIL import Image
from datasets import Dataset
import random

def load_data(image_dir, mask_dir, image_size=(224, 224)):
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

    images = np.array(images)
    masks = np.array(masks)
    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)

    dataset_dict = {
        "image": [Image.fromarray(img) for img in images],
        "label": [Image.fromarray(mask) for mask in masks],
    }

    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def get_random_example(dataset):
    img_num = random.randint(0, len(dataset) - 1)
    example_image = dataset[img_num]["image"]
    example_mask = dataset[img_num]["label"]
    return example_image, example_mask

def display_example(image, mask):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.array(image), cmap='gray')
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
