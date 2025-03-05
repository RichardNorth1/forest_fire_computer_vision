import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

__version__ = '0.1.0'

# Define dataset paths
dataset_dir = "the_wildfire_dataset_2n_version"  # Change this to your actual dataset path
img_size = (256, 256)  # Resize images to 128x128
batch_size = 32

# Data Augmentation & Loading
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

val_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "val"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

# Define a CNN model for binary classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
model.save(f'./app/model/cnn_model-{__version__}.h5')

# Evaluate on test set
test_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy:.4f}')