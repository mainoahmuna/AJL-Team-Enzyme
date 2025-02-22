import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn import preprocessing

def preprocess_data(train_df, test_df):
    label_encoder = LabelEncoder()
    train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    return train_data, val_data, label_encoder

def preprocess_image(image_path, image_size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.resize(image_size)  # Resize image
        img = np.array(img)  # Convert to numpy array
        img = img.flatten()  # Flatten the image to a 1D vector
    return img

def preprocess_images(data, image_dir, image_size=(64, 64)):
    return np.array([preprocess_image(os.path.join(image_dir, f)) for f in data['file_path']])


# 4. Logistic Regression Model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def preprocess_test_image(image_path, target_size=(64, 64)):
    """
    Load an image, resize it, normalize it, and flatten it.
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image).astype('float32') / 255.0  # Normalize
    image = image.flatten()  # Convert to 1D array
    return image

# Function to load and preprocess all images in a directory
def preprocess_images_from_directory(directory, target_size=(64, 64)):
    """
    Load all images from a directory, preprocess them, and return flattened image data with corresponding md5hashes.
    """
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    md5hashes = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.jpg')]

    images = [preprocess_test_image(image_path, target_size) for image_path in image_paths]
    
    return np.array(images), md5hashes
