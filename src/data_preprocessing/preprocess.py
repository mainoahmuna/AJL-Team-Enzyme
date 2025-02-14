import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Configuration
TRAIN_DIR = "PATH_TO_TRAIN_DIR"
TEST_DIR = "PATH_TO_TEST_DIR"
TRAIN_CSV = "PATH_TO_TRAIN_CSV"
TEST_CSV = "PATH_TO_TEST_CSV"
IMG_SIZE = (64, 64)
RANDOM_STATE = 42
METADATA_FEATURES = ['fitzpatrick_scale', 'ddi_scale']

def load_train_data(train_dir, csv_path, img_size):
    """
    Load training data with subfolders and CSV metadata

    Parameters:
    - train_dir (str): Path to the training image folder.
    - csv_path (str): Path to the CSV file containing metadata.
    - img_size (tuple): Target size for resizing images.
    """
    df = pd.read_csv(csv_path)
    df['md5hash'] = df['md5hash'].astype(str)
    metadata_dict = df.set_index('md5hash').to_dict('index')
    
    X_images = []
    X_meta = []
    y = []
    class_names = []
    
    classes = sorted([d for d in os.listdir(train_dir) 
                    if os.path.isdir(os.path.join(train_dir, d))])
    
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        print(f"Processing {class_name}...")
        
        for img_file in tqdm(os.listdir(class_path)):
            md5hash = os.path.splitext(img_file)[0]
            
            try:
                # Get metadata
                meta = metadata_dict.get(md5hash, None)
                if not meta:
                    print(f"\nMetadata missing for {md5hash}")
                    continue
                
                # Verify label consistency
                if meta['label'] != class_name:
                    print(f"\nLabel mismatch: {md5hash} CSV:{meta['label']} vs Folder:{class_name}")
                    continue
                
                # Load and process image
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path)
                img = img.convert('RGB').resize(img_size)
                img_array = np.array(img) / 255.0
                
                # Store data
                X_images.append(img_array.flatten())
                X_meta.append([meta[col] for col in METADATA_FEATURES])
                y.append(class_name)
                
                # Track unique classes
                if class_name not in class_names:
                    class_names.append(class_name)
                    
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue
    
    # Combine image and metadata features
    X_images = np.array(X_images)
    X_meta = np.array(X_meta)
    X_combined = np.hstack([X_images, X_meta])
    
    # Create class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_names))}
    y = np.array([class_to_idx[label] for label in y])
    
    return X_combined, y, class_names


def load_test_data(test_dir, csv_path, img_size):
    """
    Load test data from single folder with CSV metadata (no labels).

    Parameters:
    - test_dir (str): Path to the test image folder.
    - csv_path (str): Path to the CSV file containing metadata.
    - img_size (tuple): Target size for resizing images.

    Returns:
    - X_combined (numpy array): Combined image and metadata features.
    - img_hashes (list): List of MD5 hashes for the test images.
    """
    df = pd.read_csv(csv_path)
    df['md5hash'] = df['md5hash'].astype(str)
    metadata_dict = df.set_index('md5hash').to_dict('index')
    
    X_images = []
    X_meta = []
    img_hashes = []

    print(f"Loading test images from {test_dir}...")
    for img_file in tqdm(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_file)
        md5hash = os.path.splitext(img_file)[0]
        
        try:
            # Get metadata for the image
            meta = metadata_dict.get(md5hash, None)
            if not meta:
                print(f"\nMetadata missing for {md5hash}")
                continue

            # Load and process the image
            img = Image.open(img_path)
            img = img.convert('RGB').resize(img_size)
            img_array = np.array(img) / 255.0

            # Store the data
            X_images.append(img_array.flatten())
            X_meta.append([meta[col] for col in METADATA_FEATURES])
            img_hashes.append(md5hash)  # Store filename hash for reference
            
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            continue
    
    # Combine features
    X_images = np.array(X_images)
    X_meta = np.array(X_meta)
    X_combined = np.hstack([X_images, X_meta]) if X_meta.size else X_images
    
    return X_combined, img_hashes
