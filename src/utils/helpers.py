import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image

def plot_label_distribution(df, column="label"):
    """
    Plots the distribution of a specified label column as a horizontal bar chart.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the label column.
    column (str): The name of the label column to visualize. Default is 'label'.
    """
    value_counts = df[column].value_counts()

    plt.figure(figsize=(10, 6))
    plt.barh(value_counts.index, value_counts.values, color="skyblue")
    plt.xlabel("Count", fontsize=12)
    plt.ylabel(column.capitalize(), fontsize=12)
    plt.title(f"Distribution of {column.capitalize()}", fontsize=14)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def plot_sample_images_from_folders(data_dir, num_images=5):
    """
    Plots sample images from each class folder with labels underneath.
    
    Parameters:
    - data_dir (str): Path to the dataset directory, where each subfolder is a class.
    - num_images (int): Number of images to display per class.
    """
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    fig, axes = plt.subplots(len(class_folders), num_images, figsize=(num_images * 2, len(class_folders) * 2))
    
    for i, class_name in enumerate(class_folders):
        image_paths = glob.glob(os.path.join(data_dir, class_name, "*"))
        for j, img_path in enumerate(image_paths[:num_images]):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].set_title(class_name, fontsize=10)  # Label underneath each image
            
    plt.tight_layout()
    plt.show()