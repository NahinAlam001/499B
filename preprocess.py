import os
import shutil
import zipfile
from google.colab import drive

def mount_drive():
    drive.mount('/content/drive')

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def copy_files(source_dirs, destination_dir, file_extension):
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)

    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            subdirectory = os.path.relpath(root, source_dir)
            destination_subdirectory = os.path.join(destination_dir, subdirectory)
            os.makedirs(destination_subdirectory, exist_ok=True)

            for filename in files:
                if filename.lower().endswith(file_extension):
                    source_file = os.path.join(root, filename)
                    destination_file = os.path.join(destination_subdirectory, filename)
                    shutil.copy2(source_file, destination_file)

def main():
    # Mount Google Drive
    mount_drive()

    # Extract the zip file
    zip_path = '/content/drive/MyDrive/CSE499/CSE499.zip'
    extract_to = '/content'
    extract_zip(zip_path, extract_to)

    # Copy JPEG files (images)
    image_source_dirs = [
        "/content/ISIC-2017_Training_Data/ISIC-2017_Training_Data",
        "/content/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data",
        "/content/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data"
    ]
    image_destination_dir = "/content/isic-challenge-2017/ISIC2017_Task1-2_Training_Input"
    copy_files(image_source_dirs, image_destination_dir, ".jpg")
    print("JPEG files copied successfully!")

    # Copy PNG files (ground truth)
    ground_truth_source_dirs = [
        "/content/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth",
        "/content/ISIC-2017_Training_Part2_GroundTruth/ISIC-2017_Training_Part2_GroundTruth",
        "/content/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth",
        "/content/ISIC-2017_Test_v2_Part2_GroundTruth/ISIC-2017_Test_v2_Part2_GroundTruth",
        "/content/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth",
        "/content/ISIC-2017_Validation_Part2_GroundTruth/ISIC-2017_Validation_Part2_GroundTruth"
    ]
    ground_truth_destination_dir = "/content/isic-challenge-2017/ISIC2017_Task1-2_Training_GroundTruth"
    copy_files(ground_truth_source_dirs, ground_truth_destination_dir, ".png")
    print("PNG files copied successfully!")

if __name__ == "__main__":
    main()
