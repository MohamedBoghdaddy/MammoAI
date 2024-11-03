import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset paths for each Cervical Cancer subtype
cervical_cancer_paths = [
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_dyk',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_koc',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_mep',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_pab',
    'Multi Cancer/Multi Cancer/Cervical Cancer/cervix_sfi'
]

# ----------------------------------------
# Loading Dataset with ImageDataGenerator
# ----------------------------------------
# Create ImageDataGenerator instance with 80/20 train-validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Loop through each dataset path and create data generators
for path in cervical_cancer_paths:
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

    # Now you can proceed with model training for each cervical cancer subtype
    print(f"Data generators for {os.path.basename(path)} are ready.")
    # Here you would typically train the model with train_generator and validation_generator
