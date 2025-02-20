import os
import shutil
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def split_data(train_df, test_size=0.2, random_state=42):
    """Split the dataset into training and validation sets."""
    return train_test_split(train_df, test_size=test_size, random_state=random_state, stratify=train_df['encoded_label'])

def move_images(dataframe, source_dir, dest_dir):
    """
    Moves images into train/val directories based on labels.

    Parameters:
    - dataframe: Pandas DataFrame with 'md5hash' and 'label' columns.
    - source_dir: Directory containing labeled image folders.
    - dest_dir: Target directory for train/val split.

    Returns:
    - None
    """
    os.makedirs(dest_dir, exist_ok=True)

    for _, row in dataframe.iterrows():
        label = row['label']
        filename = row['md5hash'] + ".jpg"  # Assuming images are JPGs
        
        src_path = os.path.join(source_dir, label, filename)  # Construct source path
        label_dest_dir = os.path.join(dest_dir, label)  # Target label folder
        os.makedirs(label_dest_dir, exist_ok=True)
        
        dest_path = os.path.join(label_dest_dir, filename)

        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"Warning: File {src_path} not found!")  # Debug missing images

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

def process_data(train_csv, image_dir, train_dir, val_dir):
    """Main function to process train dataset and split images into train/val directories."""
    train_df = load_csv(train_csv)
    train_df = clean_train_data(train_df)
    train_df, label_encoder = encode_labels(train_df)
    train_data, val_data = split_data(train_df)

    # Move images based on split
    move_images(train_data, image_dir, train_dir)
    move_images(val_data, image_dir, val_dir)

    train_datagen, val_datagen = get_image_generators()
    
    train_generator = create_generator(train_dir, train_datagen)
    val_generator = create_generator(val_dir, val_datagen)

    return train_generator, val_generator, label_encoder

if __name__ == "__main__":
    TRAIN_CSV = "path to train.csv"
    IMAGE_DIR = "path to image directory"
    TRAIN_DIR = "path to processed train directory"
    VAL_DIR = "path to processed val directory"

    train_generator, val_generator, label_encoder = process_data(TRAIN_CSV, IMAGE_DIR, TRAIN_DIR, VAL_DIR)