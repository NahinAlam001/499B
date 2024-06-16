import os
import pandas as pd
from shutil import copy2

# Define the paths
base_dir = "Dataset"
images_dir = os.path.join(base_dir, "images")
csv_file = "data.csv"
output_dir = os.path.join(base_dir, "ImageFolder")

# Read the CSV file
df = pd.read_csv(csv_file)

# Create class directories
class_names = df['dx'].unique()
for class_name in class_names:
    class_dir = os.path.join(output_dir, class_name.replace(' ', '_'))  # Replace spaces with underscores for directory names
    os.makedirs(class_dir, exist_ok=True)

# Copy images to their respective class directories
for idx, row in df.iterrows():
    image_id = row['image_id']
    class_name = row['dx'].replace(' ', '_')  # Replace spaces with underscores for directory names
    image_name = f"{image_id}.bmp"
    src_image_path = os.path.join(images_dir, image_name)
    dest_image_path = os.path.join(output_dir, class_name, image_name)
    if os.path.exists(src_image_path):
        copy2(src_image_path, dest_image_path)
    else:
        print(f"Warning: {src_image_path} does not exist.")

print("Images have been organized into the ImageFolder structure.")
