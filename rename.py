import os

# Define the paths to the 'images' and 'masks' folders
images_folder = "./../Downloads/images"
masks_folder = "./../Downloads/masks"

# Get the list of image and mask filenames
image_filenames = sorted(os.listdir(images_folder))
mask_filenames = sorted(os.listdir(masks_folder))

# Iterate through the filenames and rename the mask files
for image_filename, mask_filename in zip(image_filenames, mask_filenames):
    # Construct the full paths to the image and mask files
    image_path = os.path.join(images_folder, image_filename)
    mask_path = os.path.join(masks_folder, mask_filename)

    # Extract the filename without extension from the image filename
    image_name_without_ext = os.path.splitext(image_filename)[0]

    # Construct the new mask filename with the "_mask" suffix and ".jpg" extension
    mask_new_filename = f"{image_name_without_ext}_mask.jpg"
    mask_new_path = os.path.join(masks_folder, mask_new_filename)

    # Rename the mask file
    os.rename(mask_path, mask_new_path)
    print(f"Renamed {mask_filename} to {mask_new_filename}")
