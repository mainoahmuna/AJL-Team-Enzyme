import os
import pandas as pd
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tqdm import tqdm

def load_csv(file_path):
    """Load a CSV file into a Pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def clean_train_data(train_df):
    """Clean the training dataset by dropping unnecessary columns and handling missing values."""
    cols_to_drop = ['qc', 'nine_partition_label', 'three_partition_label']
    train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns], inplace=True)
    train_df.dropna(inplace=True)
    return train_df

def encode_labels(train_df):
    """Encode categorical labels into numerical values."""
    if 'label' not in train_df.columns:
        raise ValueError("Column 'label' not found in train dataset.")
    
    label_encoder = LabelEncoder()
    train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
    return train_df, label_encoder

def get_image_generators():
    """Returns image data generators for training and validation."""
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, val_datagen

def create_generator(directory, datagen, batch_size=32, target_size=(128, 128)):
    """
    Creates an image generator for training or validation.

    Parameters:
    - directory: Path to the dataset directory where each subdirectory is a label.
    - datagen: An instance of ImageDataGenerator.
    - batch_size: Number of images per batch.
    - target_size: Target size for image resizing.

    Returns:
    - A generator for model training/validation.
    """
    if not isinstance(datagen, ImageDataGenerator):
        raise TypeError("datagen must be an instance of ImageDataGenerator")
    
    generator = datagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",  # One-hot encoding for labels
        shuffle=True
    )
    return generator

def load_train_data(train_csv, image_dir, train_dir, val_dir, img_size=(128, 128)):
    """Main function to process train dataset and split images into train/val directories."""
    # Load CSV metadata
    train_df = load_csv(train_csv)
    train_df = clean_train_data(train_df)
    
    # Encode labels
    train_df, label_encoder = encode_labels(train_df)

    # Add .jpg extension to the md5hash column
    train_df['md5hash'] = train_df['md5hash'].astype(str) + '.jpg'

    # Combine label and md5hash to form file path
    train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash']

    # Move images based on labels
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for _, row in train_df.iterrows():
        label = row['label']
        file_path = row['file_path']
        img_path = os.path.join(image_dir, file_path)  # Path to the image
        
        # Move image to appropriate train/val folder
        if np.random.rand() < 0.8:  # 80% for training
            dest_path = os.path.join(train_dir, label)
        else:  # 20% for validation
            dest_path = os.path.join(val_dir, label)
        
        os.makedirs(dest_path, exist_ok=True)
        dest_file = os.path.join(dest_path, row['md5hash'])
        
        if os.path.exists(img_path):
            shutil.copy(img_path, dest_file)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    # Get image generators for train/val
    train_datagen, val_datagen = get_image_generators()

    train_generator = create_generator(train_dir, train_datagen, batch_size=32, target_size=img_size)
    val_generator = create_generator(val_dir, val_datagen, batch_size=32, target_size=img_size)

    return train_generator, val_generator, label_encoder

if __name__ == "__main__":
    TRAIN_CSV = "path to train.csv"
    IMAGE_DIR = "path to image directory"
    TRAIN_DIR = "path to processed train directory"
    VAL_DIR = "path to processed val directory"

    # Process data and get generators
    train_generator, val_generator, label_encoder = load_train_data(TRAIN_CSV, IMAGE_DIR, TRAIN_DIR, VAL_DIR)