mport os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define SAM optimizer class
class SAMOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, **kwargs):
        super(SAMOptimizer, self).__init__(**kwargs)
        self.base_optimizer = base_optimizer
        self.rho = rho

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'shadow')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr = apply_state[(var, self.base_optimizer.learning_rate)]
        wd = apply_state[(var, self.base_optimizer.weight_decay)]
        lr_t = lr / (1 - wd)

        shadow = self.get_slot(var, 'shadow')
        var_t = var + self.rho * (shadow - var)
        grad_norm = tf.norm(grad)
        shadow.assign(var_t - lr_t * grad)

        return self.base_optimizer._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(SAMOptimizer, self).get_config()
        config.update({
            'base_optimizer': tf.keras.optimizers.serialize(self.base_optimizer),
            'rho': self.rho
        })
        return config

# Function to create CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

# Path to your dataset
dataset_path = '/path/to/your/data/ISIC_2019_Training_Input'

# Load and prepare data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

# Create the model
model = create_model()

# Define SAM optimizer
base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
sam_optimizer = SAMOptimizer(base_optimizer, rho=0.05)

# Compile the model
model.compile(optimizer=sam_optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10)

# Plot training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
