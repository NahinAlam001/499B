import os
import zipfile
import subprocess

def clone_repo():
    subprocess.run(["git", "clone", "-b", "SAM", "https://github.com/NahinAlam001/499B.git"])

def download_dataset():
    subprocess.run(["gdown", "https://drive.google.com/uc?id=1FomR78IlcqHqNi77fXAJCMgvEebihhF-"])

def extract_dataset(zip_file, extract_to="."):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    # Clone the repository
    clone_repo()
    
    # Change directory to the cloned repository
    os.chdir('499B')

    # Download the dataset
    download_dataset()

    # Extract the dataset
    extract_dataset('Dataset.zip')
