# SAM (Segment Anything Model) - Modular Code

This repository contains a modular implementation of the Segment Anything Model (SAM) for segmenting images from the PH2 dataset. The implementation includes scripts for data preprocessing, training, and evaluation of the model.

## Table of Contents
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Metrics](#metrics)

## Installation

Before using the scripts, you need to install the necessary dependencies. You can install them using the `install_dependencies.py` script provided in the repository.

1. Clone the repository:
    ```sh
    git clone -b SAM https://github.com/NahinAlam001/499B.git
    cd SAM
    ```

2. Run the dependency installation script:
    ```sh
    python install_dependencies.py
    ```

This will install all required packages including `monai`, `datasets`, `scikit-learn`, `fvcore`, and specific packages from GitHub.

## Data Preprocessing

The data preprocessing script downloads the dataset, extracts the dataset, and organizes the image and ground truth files into appropriate directories.

1. Run the preprocessing script:
    ```sh
    python preprocess.py
    ```

This script will:
- Download the dataset
- Extract the PH2 dataset from a zip file.

## Training

The training script trains the SAM model on the PH2 dataset and saves the trained model checkpoint.

1. Run the training script:
    ```sh
    python train.py
    ```

This script will:
- Load and preprocess the data.
- Train the SAM model.
- Save the model checkpoint to `skin_model_PH2_SAM_checkpoint.pth`.

## Evaluation

The evaluation script evaluates the trained SAM model on the PH2 dataset and calculates various performance metrics.

1. Run the evaluation script:
    ```sh
    python evaluate.py
    ```

This script will:
- Load the trained model and the evaluation dataset.
- Evaluate the model and print the following metrics:
  - Mean Jaccard Score
  - Mean Dice Score
  - Mean Accuracy
  - Mean Recall
  - Mean F1 Score

## Metrics

The evaluation script calculates and prints the following metrics:
- **Mean Jaccard Score**: Intersection over Union (IoU) score.
- **Mean Dice Score**: Dice coefficient, a measure of overlap between predicted and ground truth masks.
- **Mean Accuracy**: The ratio of correctly predicted pixels to total pixels.
- **Mean Recall**: The ratio of correctly predicted positive pixels to all ground truth positive pixels.
- **Mean F1 Score**: The harmonic mean of precision and recall.

## Directory Structure

The directory structure after running the preprocessing script should look like this:

```
/499B
  ├── Dataset
      ├── images
      ├── masks
```

## Notes

- Ensure you have access to Google Drive and the dataset is correctly placed in your Drive.
- Modify paths in the scripts if your directory structure is different.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/499B/SAM/issues).

---

By following this guide, you should be able to set up, train, and evaluate the SAM model on the PH2 dataset. This modular approach allows for easier maintenance and scalability of the project.
