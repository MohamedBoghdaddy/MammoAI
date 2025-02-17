import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define dataset paths for each Cervical Cancer subtype
cervical_cancer_paths = [
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_dyk',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_koc',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_mep',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_pab',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_sfi'
]

# Create ImageDataGenerator instance with 80/20 train-validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generators = []
validation_generators = []

for path in cervical_cancer_paths:
    train_generator = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    train_generators.append(train_generator)
    validation_generators.append(validation_generator)
    
    print(f"Data generators for {os.path.basename(path)} are ready.")

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
        Dense(len(cervical_cancer_paths), activation='softmax')
    ])
    return model

# Compile model
model = build_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
for i, path in enumerate(cervical_cancer_paths):
    print(f"Training model for dataset: {os.path.basename(path)}")
    model.fit(
        train_generators[i],
        validation_data=validation_generators[i],
        epochs=10,
        verbose=1
    )

# Save model
model.save("cervical_cancer_classification_model.h5")

print("Model training complete and saved as 'cervical_cancer_classification_model.h5'")
