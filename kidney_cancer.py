import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path for Kidney Cancer
dataset_path = 'Multi Cancer/Multi Cancer/Kidney Cancer/kidney_tumor'

# ImageDataGenerator with 80/20 train-validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Train data generator
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Data generators for Kidney Cancer are ready.")
