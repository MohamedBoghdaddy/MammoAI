import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset paths for Lung and Colon Cancer
dataset_paths = [
    'Multi Cancer/Multi Cancer/Lung and Colon Cancer/lung_adenocarcinoma',
    'Multi Cancer/Multi Cancer/Lung and Colon Cancer/colon_adenocarcinoma'
]

# ImageDataGenerator with 80/20 train-validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

for path in dataset_paths:
    # Train data generator
    train_generator = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # Validation data generator
    validation_generator = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    print(f"Data generators for {os.path.basename(path)} (Lung and Colon Cancer) are ready.")
