from flask import Flask, request, render_template, redirect, url_for
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
from transformers import SamModel, SamImageProcessor
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3  # Update this to the actual number of classes
checkpoint_path = "checkpoint.pth"

class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        return x

classifier = DenseNetClassifier(num_classes=num_classes)
checkpoint = torch.load(checkpoint_path, map_location=device)
classifier.load_state_dict(checkpoint['classifier_state_dict'])
classifier = classifier.to(device)
classifier.eval()

image_processor = SamImageProcessor()
sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_model = sam_model.to(device)
sam_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
    else:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    return [x_min, y_min, x_max, y_max]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for('predict', filepath=filepath))
    return render_template('index.html', title='Home')

@app.route('/predict')
def predict():
    filepath = request.args.get('filepath')
    if filepath:
        image = cv2.imread(filepath)
        original_size = image.shape[:2]
        print(f"Original image size: {original_size}")

        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Transformed image tensor shape: {image_tensor.shape}")

        with torch.no_grad():
            sam_output = sam_model(image_tensor, input_boxes=None)
            pred_masks = sam_output["pred_masks"]
            print(f"Predicted masks shape: {pred_masks.shape}")

            pred_masks = pred_masks.squeeze(1)
            pred_masks = pred_masks.permute(0, 2, 3, 1).reshape(pred_masks.shape[0], pred_masks.shape[2], pred_masks.shape[3], -1)
            pred_masks = pred_masks.permute(0, 3, 1, 2)
            pred_masks = pred_masks[:, :3, :, :]
            pred_masks = torch.nn.functional.interpolate(pred_masks, size=(256, 256), mode='bilinear')
            print(f"Interpolated masks shape: {pred_masks.shape}")

            image_tensor = torch.nn.functional.interpolate(image_tensor, size=(256, 256), mode='bilinear')
            print(f"Interpolated image tensor shape: {image_tensor.shape}")

            combined_input = torch.cat((image_tensor, pred_masks), dim=1)
            print(f"Combined input shape: {combined_input.shape}")

            outputs = classifier(combined_input)
            print(f"Model outputs: {outputs}")

            _, predicted = torch.max(outputs, 1)
            print(f"Predicted class index: {predicted.item()}")

        # Define the label map
        label_map = {0: 'Class1', 1: 'Class2', 2: 'Common Nevus'}

        # Map the prediction to the class label
        predicted_label = label_map.get(predicted.item(), "Unknown class")

        return render_template('result.html', title='Prediction Result', predicted_label=predicted_label)
    return "No filepath provided."

if __name__ == '__main__':
    app.run(debug=True)
