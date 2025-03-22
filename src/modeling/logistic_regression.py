<<<<<<< HEAD
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from tqdm import tqdm

# %%
# Configuration
TRAIN_DIR = "path to training image directory"
TEST_DIR = "path to testing image directory"
TRAIN_CSV = "path to training CSV file"
TEST_CSV = "path to testing CSV file"
IMG_SIZE = (64, 64)
RANDOM_STATE = 42
METADATA_FEATURES = ['fitzpatrick_scale', 'ddi_scale', 'fitzpatrick_centaur']

# %%
def load_train_data(train_dir, csv_path, img_size):
    """Load training data with subfolders and CSV metadata"""
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

# %%
def load_test_data(test_dir, csv_path, img_size, class_names):
    """Load test data from single folder with CSV labels"""
    df = pd.read_csv(csv_path)
    df['md5hash'] = df['md5hash'].astype(str)
    metadata_dict = df.set_index('md5hash').to_dict('index')
    
    X_images = []
    X_meta = []
    y = []
    
    print(f"Loading test images from {test_dir}...")
    for img_file in tqdm(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_file)
        md5hash = os.path.splitext(img_file)[0]
        
        try:
            meta = metadata_dict.get(md5hash, None)
            if not meta:
                print(f"\nMetadata missing for {md5hash}")
                continue
                
            label = meta.get('label', None)
            if label not in class_names:
                print(f"\nSkipping {md5hash} - unknown label {label}")
                continue
                
            # Load and process image
            img = Image.open(img_path)
            img = img.convert('RGB').resize(img_size)
            img_array = np.array(img) / 255.0
            
            # Store data
            X_images.append(img_array.flatten())
            X_meta.append([meta[col] for col in METADATA_FEATURES])
            y.append(class_names.index(label))
            
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            continue
    
    # Combine features
    X_images = np.array(X_images)
    X_meta = np.array(X_meta)
    X_combined = np.hstack([X_images, X_meta]) if X_meta.size else X_images
    
    return X_combined, np.array(y)

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, class_names, fitzpatrick_scale):
    """
    Evaluate a trained model on the test set, showcasing performance across different Fitzpatrick Scale categories.

    Parameters:
        model (sklearn model): The trained machine learning model.
        X_test (DataFrame or ndarray): Test data.
        y_test (ndarray): True test labels.
        class_names (list): List of class labels.
        fitzpatrick_scale (ndarray or Series): Fitzpatrick Scale values for each sample in X_test.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print(f"\nEvaluating {model.__class__.__name__}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute overall evaluation metrics
    overall_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    # Print overall performance
    print(f"\nOverall Model Performance:")
    for key, value in overall_metrics.items():
        if key != "conf_matrix":
            print(f"{key.capitalize()}: {value:.4f}")
    print("\nConfusion Matrix:\n", overall_metrics["conf_matrix"])
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

    # Analyze performance across Fitzpatrick Scale categories
    unique_scales = np.unique(fitzpatrick_scale)
    scale_metrics = {}

    print("\nPerformance by Fitzpatrick Scale:")
    for scale in unique_scales:
        mask = fitzpatrick_scale == scale
        if np.sum(mask) == 0:
            continue  # Skip if no samples for this scale
        
        y_test_scale = y_test[mask]
        y_pred_scale = y_pred[mask]

        scale_metrics[scale] = {
            "accuracy": accuracy_score(y_test_scale, y_pred_scale),
            "precision": precision_score(y_test_scale, y_pred_scale, average="weighted", zero_division=0),
            "recall": recall_score(y_test_scale, y_pred_scale, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test_scale, y_pred_scale, average="weighted", zero_division=0),
        }

        print(f"\nFitzpatrick Scale {scale}:")
        for key, value in scale_metrics[scale].items():
            print(f"{key.capitalize()}: {value:.4f}")

    return {"overall": overall_metrics, "fitzpatrick_scale_metrics": scale_metrics}

# %%
# Load data
X, y, class_names = load_train_data(TRAIN_DIR, TRAIN_CSV, IMG_SIZE)


# %%


# %%



=======
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
>>>>>>> main
