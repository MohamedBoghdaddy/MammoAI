# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tkinter import filedialog, Label, Button
import tkinter as tk
from PIL import ImageTk, Image

# -----------------------------------------------------------
# Part 1: CNN Model for Image-Based Prediction
# -----------------------------------------------------------

# Build a CNN model
def build_cnn_model():
    model = Sequential()

    # Convolutional layer and max pooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Additional layers
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load pre-trained weights if available, else train the model on a dataset.
cnn_model = build_cnn_model()
cnn_model.load_weights('model_weights.h5')  # Load pre-trained weights if available

# Function to preprocess and predict image
def preprocess_and_predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = cnn_model.predict(img)
    return 'Cancer Detected' if prediction[0][0] > 0.5 else 'No Cancer Detected'

# GUI Implementation using Tkinter
def upload_image():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    
    panel = Label(window, image=img)
    panel.image = img
    panel.grid(row=2, column=0, columnspan=2)

    result = preprocess_and_predict(file_path)
    result_label.config(text=result)

# -----------------------------------------------------------
# Part 2: Logistic Regression Model for CSV-Based Prediction
# -----------------------------------------------------------

# Load the dataset from the CSV file
df = pd.read_csv('breast-cancer-dataset.csv')

# Display first few rows to understand the structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Define features (X) and target (y)
X = df.drop('target', axis=1)  # Drop the target column to keep features only
y = df['target']  # Define the target column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# -----------------------------------------------------------
# Part 3: Tkinter GUI for Image Upload
# -----------------------------------------------------------

# Tkinter GUI setup
window = tk.Tk()
window.title("MammoAI - Breast Cancer Detection")

label = Label(window, text="Upload a Mammogram Image", font=("Arial", 14))
label.grid(row=0, column=0, columnspan=2)

upload_button = Button(window, text="Upload Image", command=upload_image)
upload_button.grid(row=1, column=0, columnspan=2)

result_label = Label(window, text="", font=("Arial", 14), fg="blue")
result_label.grid(row=3, column=0, columnspan=2)

# Start the GUI loop
window.mainloop()
