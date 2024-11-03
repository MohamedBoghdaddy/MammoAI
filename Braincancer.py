import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path directly
dataset_path = 'Multi Cancer/Multi Cancer/Brain Cancer/brain_glioma'
dataset_path = 'Multi Cancer/Multi Cancer/Brain Cancer/brain_menin'
dataset_path = 'Multi Cancer/Multi Cancer/Brain Cancer/brain_tumor'


# ----------------------------------------
# Loading Dataset with ImageDataGenerator
# ----------------------------------------
# Create ImageDataGenerator instance
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80/20 train-validation split

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

# Now you can proceed with model training, using `train_generator` and `validation_generator`
