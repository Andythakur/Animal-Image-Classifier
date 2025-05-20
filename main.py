#  Import all required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# step 1: set image size and path to dataset
image_size = 224
batch_size = 32
data_dir = os.path.join(os.getcwd(),"dataset")  # Points to 'dataset' folder in your project

#step 2: prepare the image data with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel value
    validation_split=0.2,     # 20% data used for validation
    horizontal_flip=True,    # flip i mage to improve tarining
    zoom_range=0.2           # Slight zooming
)

# Load training data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validastion data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Load a pre-trained model (Transfer Learning)
base_model = MobileNetV2(input_shape=(image_size, image_size, 3),
                         include_top=False,     # Don't use original classification
                         weights='imagenet')
base_model.trainable = False      #Freze the base model

# Step 4: Add custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(15, activation='softmax')(x)  # 15 classes

#   Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Step 5: Compile the model (MISSING STEP FIXE)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step6: Train the mode
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10 # You can change this number
)

# Step 7: Plot tarining vs validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# step 8: Save the Training Model
model.save("animal_model.h5")
print("Model training complete and saved as animal_model.h5")