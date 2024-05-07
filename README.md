**CNN Model with Batch Normalization**

- **Architecture Overview**:
  - Input consists of 3-channel images (RGB).
  - Three convolutional layers are used, each followed by batch normalization, ReLU activation, and max-pooling.
  - Two fully connected layers are used for classification, with dropout regularization applied to the first fully connected layer.

- **Convolutional Layers**:
  - `self.conv1`: First convolutional layer with 3 input channels, 16 output channels, kernel size of 3x3, and 'same' padding.
  - `self.conv2`: Second convolutional layer with 16 input channels, 32 output channels, kernel size of 3x3, and 'same' padding.
  - `self.conv3`: Third convolutional layer with 32 input channels, 64 output channels, kernel size of 3x3, and 'same' padding.

- **Batch Normalization Layers**:
  - `self.bn1`: Batch normalization layer after the first convolutional layer.
  - `self.bn2`: Batch normalization layer after the second convolutional layer.
  - `self.bn3`: Batch normalization layer after the third convolutional layer.

- **Pooling Layers**:
  - `self.pool1`: Max-pooling layer with a kernel size of 2x2 and a stride of 2 after the first convolutional layer.
  - `self.pool2`: Max-pooling layer with a kernel size of 2x2 and a stride of 2 after the second convolutional layer.
  - `self.pool3`: Max-pooling layer with a kernel size of 2x2 and a stride of 2 after the third convolutional layer.

- **Fully Connected Layers**:
  - `self.fc1`: First fully connected layer with 64 * 8 * 8 input features (resulting from the last convolutional layer), 512 output features.
  - `self.fc2`: Second fully connected layer with 512 input features and `num_classes` output features, representing class scores.

- **Dropout**:
  - `self.dropout1`: Dropout layer with a dropout probability of 0.5 applied after the first fully connected layer to reduce overfitting during training.

- **Forward Pass**:
  - Input images pass through convolutional layers with batch normalization and ReLU activation, followed by max-pooling.
  - After flattening, the features are passed through fully connected layers with ReLU activation and dropout.
  - Output logits are obtained from the last fully connected layer, representing class predictions.

This architecture aims to improve the training stability and convergence speed of the CNN by using batch normalization after each convolutional layer. It also incorporates dropout regularization to prevent overfitting, especially in deeper layers of the network.
