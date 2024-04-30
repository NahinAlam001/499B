import random
from pathlib import Path

def ph2(dataset_dir, output_dir, image_ext='bmp', mask_ext='bmp',
        train_ratio=0.7, val_ratio=0.1, split_random=True, random_seed=None):
  if random_seed is not None:
    random.seed(random_seed)
  if train_ratio + val_ratio > 1:
    raise ValueError("train_ration + val_ratio is too big")

  dataset_dir = Path(dataset_dir)

  train_val_test_images = []
  train_val_test_masks = []

  for tmp_path in (dataset_dir / 'PH2 Dataset images').iterdir():
    train_val_test_images.append(
      tmp_path / ("%s_Dermoscopic_Image" % tmp_path.name) / f"{tmp_path.name}.{image_ext}")
    train_val_test_masks.append(tmp_path / ("%s_lesion" % tmp_path.name) / f"{tmp_path.name}_lesion.{mask_ext}")

  train_val_test_data = list(zip(train_val_test_images, train_val_test_masks))

  train_count = int(len(train_val_test_data) * train_ratio)
  val_count = int(len(train_val_test_data) * val_ratio)
  # test_count = len(train_val_test_data) - train_count - val_count

  if split_random:
    random.shuffle(train_val_test_data)

  train_data = train_val_test_data[:train_count]
  val_data = train_val_test_data[train_count:train_count + val_count]
  test_data = train_val_test_data[train_count + val_count:]

  __save_output(output_dir, train_data, val_data, test_data)

def __save_output(output_dir, train_data, val_data, test_data):
  output_dir = Path(output_dir)

  output_dir.mkdir(parents=True, exist_ok=True)

  if len(train_data) != 0:
    fp_images = open(output_dir.joinpath("train_images.txt"), 'w')
    fp_masks = open(output_dir.joinpath("train_masks.txt"), 'w')

    for item in train_data:
      fp_images.write("%s\n" % item[0])
      fp_masks.write("%s\n" % item[1])

    fp_images.close()
    fp_masks.close()

  if len(val_data) != 0:
    fp_images = open(output_dir.joinpath("val_images.txt"), 'w')
    fp_masks = open(output_dir.joinpath("val_masks.txt"), 'w')

    for item in val_data:
      fp_images.write("%s\n" % item[0])
      fp_masks.write("%s\n" % item[1])

    fp_images.close()
    fp_masks.close()

  if len(test_data) != 0:
    fp_images = open(output_dir.joinpath("test_images.txt"), 'w')
    fp_masks = open(output_dir.joinpath("test_masks.txt"), 'w')

    for item in test_data:
      fp_images.write("%s\n" % item[0])
      fp_masks.write("%s\n" % item[1])

    fp_images.close()
    fp_masks.close()

if __name__ == '__main__':
  random_seed = 1234
  ph2(dataset_dir="datasets/PH2", output_dir="data/ph2", random_seed=random_seed)
