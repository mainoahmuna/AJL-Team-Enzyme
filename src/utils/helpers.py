import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image
from tqdm import tqdm

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

def get_pixel_statistics(train_dir):
    # Initialize variables for RGB channels
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_r, sum_sq_g, sum_sq_b = 0.0, 0.0, 0.0
    min_r = min_g = min_b = 255.0
    max_r = max_g = max_b = 0.0
    total_pixels = 0
    
    # Iterate through class directories
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Process images with progress bar
        for img_file in tqdm(os.listdir(class_dir), 
                          desc=f'Processing {class_name}', 
                          unit='img'):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_array = np.array(img)
                    
                    h, w, _ = img_array.shape
                    num_pixels = h * w
                    total_pixels += num_pixels
                    
                    for c in range(3):
                        channel = img_array[..., c]
                        sum_ = np.sum(channel)
                        sum_sq = np.sum(channel ** 2)
                        curr_min = np.min(channel)
                        curr_max = np.max(channel)
                        
                        if c == 0:  # Red
                            sum_r += sum_
                            sum_sq_r += sum_sq
                            min_r = min(min_r, curr_min)
                            max_r = max(max_r, curr_max)
                        elif c == 1:  # Green
                            sum_g += sum_
                            sum_sq_g += sum_sq
                            min_g = min(min_g, curr_min)
                            max_g = max(max_g, curr_max)
                        else:  # Blue
                            sum_b += sum_
                            sum_sq_b += sum_sq
                            min_b = min(min_b, curr_min)
                            max_b = max(max_b, curr_max)
                            
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue

    # Calculate final statistics
    def get_stats(sum_, sum_sq, total, min_, max_):
        mean = sum_ / total
        variance = (sum_sq / total) - (mean ** 2)
        variance = max(variance, 0.0)
        std = np.sqrt(variance)
        return mean, std, min_, max_

    # Missing: Calculate stats for each channel
    stats_r = get_stats(sum_r, sum_sq_r, total_pixels, min_r, max_r)
    stats_g = get_stats(sum_g, sum_sq_g, total_pixels, min_g, max_g)
    stats_b = get_stats(sum_b, sum_sq_b, total_pixels, min_b, max_b)

    return {
        'red': {'mean': stats_r[0], 'std': stats_r[1], 
                'min': stats_r[2], 'max': stats_r[3]},
        'green': {'mean': stats_g[0], 'std': stats_g[1], 
                 'min': stats_g[2], 'max': stats_g[3]},
        'blue': {'mean': stats_b[0], 'std': stats_b[1], 
                'min': stats_b[2], 'max': stats_b[3]}
    }