import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define paths for the three datasets
dataset_1_dir = '/home/saivarun/AD/AD dataset'  # e.g., Alzheimer’s Dataset 1
dataset_2_dir = '/home/saivarun/AD/Combined Dataset'  # e.g., Alzheimer’s Dataset 2
dataset_3_dir = '/home/saivarun/AD/AugmentedAlzheimerDataset'  # New third dataset path
dataset_4_dir = '/home/saivarun/AD/alzheimer_s dataset' 

# Define the combined dataset output path
combined_train_dir = '/home/saivarun/AD/final_dataset/train'
combined_test_dir = '/home/saivarun/AD/final_dataset/test'

# Define the target image size (e.g., 224x224 for most deep learning models)
target_size = (224, 224)

# Create directories for the combined dataset if they don't exist
os.makedirs(combined_train_dir, exist_ok=True)
os.makedirs(combined_test_dir, exist_ok=True)

# Ensure the labels are consistent across all datasets
# Combine the possible labels from all datasets
class_labels = ['Healthy', 'Alzheimer', 'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Function to check and create labels in the output directories
def create_class_dirs(directory):
    for label in class_labels:
        os.makedirs(os.path.join(directory, label), exist_ok=True)

# Function to resize and save images to the new directory
def resize_and_save(image_path, label, output_dir):
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Resize the image
        image_resized = cv2.resize(image, target_size)

        # Convert image to a format that can be saved (optional, but useful for some models)
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Create the path for the label class directory in the output directory
        label_dir = os.path.join(output_dir, label)

        # Copy the image to the new location
        filename = os.path.basename(image_path)
        save_path = os.path.join(label_dir, filename)
        
        cv2.imwrite(save_path, image_resized)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Function to copy images from a dataset to the combined directory (for datasets with train/test directories)
def copy_images_from_dataset_with_train_test(dataset_dir, output_train_dir, output_test_dir):
    for label in class_labels:
        for split in ['train', 'test']:  # We need to process both train and test splits
            label_dir = os.path.join(dataset_dir, split, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    if os.path.isfile(image_path) and image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        if split == 'train':
                            resize_and_save(image_path, label, output_train_dir)
                        else:
                            resize_and_save(image_path, label, output_test_dir)

# Function to copy and split images for the third dataset (which doesn't have train/test splits)
def copy_and_split_images_from_dataset(dataset_dir, output_train_dir, output_test_dir):
    for label in class_labels:
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            image_paths = [os.path.join(label_dir, image_name) for image_name in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, image_name)) and image_name.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Split into train/test using train_test_split
            train_paths, test_paths = train_test_split(image_paths, test_size=0.2)
            
            # Copy to train directory
            for image_path in train_paths:
                resize_and_save(image_path, label, output_train_dir)
            
            # Copy to test directory
            for image_path in test_paths:
                resize_and_save(image_path, label, output_test_dir)

# Create class directories in the combined training and testing directories
create_class_dirs(combined_train_dir)
create_class_dirs(combined_test_dir)

# Copy images from the first dataset (train and test directories)
copy_images_from_dataset_with_train_test(dataset_1_dir, combined_train_dir, combined_test_dir)

# Copy images from the second dataset (train and test directories)
copy_images_from_dataset_with_train_test(dataset_2_dir, combined_train_dir, combined_test_dir)

# Copy and split images from the third dataset (no train/test directories)
copy_and_split_images_from_dataset(dataset_3_dir, combined_train_dir, combined_test_dir)
# Copy images from the second dataset (train and test directories)
copy_images_from_dataset_with_train_test(dataset_4_dir, combined_train_dir, combined_test_dir)

print("Datasets aggregated, resized, and split into train/test sets.")
