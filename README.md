# UNet Image Segmentation

This project implements an image segmentation model using the UNet architecture in PyTorch. The model is trained and evaluated on the ISIC 2018 challenge dataset for skin lesion segmentation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Metrics](#metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project focuses on building a UNet model for the segmentation of skin lesions. The UNet model is a popular architecture for image segmentation tasks due to its ability to capture both spatial and context information through its encoder-decoder structure.

## Dataset
The dataset used in this project is the ISIC 2018 challenge dataset, which can be downloaded from the [ISIC Archive](https://challenge2018.isic-archive.com/). The dataset consists of dermoscopic images of skin lesions along with their corresponding segmentation masks.

## Requirements
To install the required packages, run:
```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the following dependencies:
```
numpy==1.22.4
opencv-python==4.5.5.64
pandas==1.4.2
tqdm==4.64.0
scikit-learn==1.1.1
torch==1.12.0
torchvision==0.13.0
```

## Usage

### Training
To train the UNet model, run the `train.py` script:
```bash
python train.py --dataset_path /path/to/your/dataset/ --batch_size 4 --lr 0.0001 --epochs 5 --model_path files/model.pth
```

### Evaluation
To evaluate the trained model, run the `eval.py` script:
```bash
python eval.py --dataset_path /path/to/your/dataset/ --model_path files/model.pth
```

## Model Architecture
The UNet model consists of an encoder and a decoder with skip connections. The encoder captures context information by downsampling the input image, while the decoder restores the spatial dimensions through upsampling.

### UNet Implementation (`model.py`)
```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.downsample(enc1))
        enc3 = self.encoder3(self.downsample(enc2))
        enc4 = self.encoder4(self.downsample(enc3))

        bottleneck = self.bottleneck(self.downsample(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

    def downsample(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)
```

## Metrics
The model is evaluated using several metrics:
- **Dice Coefficient**: Measures the overlap between the predicted and ground truth masks.
- **Intersection over Union (IoU)**: Measures the intersection over union between the predicted and ground truth masks.
- **Accuracy**: Measures the overall pixel-wise accuracy.
- **Precision**: Measures the ratio of true positives to the sum of true positives and false positives.
- **Recall**: Measures the ratio of true positives to the sum of true positives and false negatives.
- **F1 Score**: The harmonic mean of precision and recall.

## Results
Evaluation metrics are saved to a CSV file (`files/score.csv`). Example results might look like:
```
Image Name, Acc, F1, Jaccard, Recall, Precision
ISIC_001.jpg, 0.95, 0.89, 0.80, 0.91, 0.87
...
```

## Contributing
Contributions are welcome! If you have suggestions for improvements, please create a pull request or open an issue.

## License
This project is licensed under the MIT License.
```

### Instructions to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/unet-image-segmentation.git
   cd unet-image-segmentation
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - Download the ISIC 2018 dataset and place it in the specified directory.

4. **Train the Model**:
   ```bash
   python train.py --dataset_path /path/to/your/dataset/ --batch_size 4 --lr 0.0001 --epochs 5 --model_path files/model.pth
   ```

5. **Evaluate the Model**:
   ```bash
   python eval.py --dataset_path /path/to/your/dataset/ --model_path files/model.pth
   ```

Replace `/path/to/your/dataset/` with the actual path where your ISIC 2018 dataset is stored.
