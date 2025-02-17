import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define dataset path for Kidney Cancer
dataset_path = 'Multi Cancer/Multi Cancer/Kidney Cancer/kidney_tumor'

# ImageDataGenerator with 80/20 train-validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

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

# Define CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    return model

# Compile model
model = build_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training model for Kidney Cancer classification...")
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

# Save model
model.save("kidney_cancer_classification_model.h5")

print("Model training complete and saved as 'kidney_cancer_classification_model.h5'")
