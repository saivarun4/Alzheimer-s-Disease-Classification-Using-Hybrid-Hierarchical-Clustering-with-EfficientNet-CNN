import os
import shutil
import random

# Define paths for the train and test directories
train_dir = '/home/saivarun/AD/final_dataset/train'
test_dir = '/home/saivarun/AD/final_dataset/test'

# Define paths for the new unbalanced directories
balanced_train_dir = '/home/saivarun/AD/balanced_dataset/train'
balanced_test_dir = '/home/saivarun/AD/balanced_dataset/test'

# Define class labels
class_labels = ['NonDemented', 'ModerateDemented', 'VeryMildDemented', 'MildDemented']

# Function to unbalance the dataset by undersampling (removing images) for train set
def unbalance_train_dataset(input_dir, output_dir, reduction_factor=0.5):
    """
    Function to unbalance the dataset by reducing the number of images in each class for the train dataset.
    
    Args:
    - input_dir: Directory containing the input data.
    - output_dir: Directory to store the unbalanced data.
    - reduction_factor: The factor by which to reduce the dataset size (default is 0.5, i.e., 50%).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each class for the training dataset
    for label in class_labels:
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        
        # Make sure the output directory for each label exists
        os.makedirs(output_label_dir, exist_ok=True)
        
        # Check if the label directory exists
        if not os.path.exists(label_dir):
            print(f"[ERROR] Directory for label '{label}' does not exist: {label_dir}")
            continue
        
        # Get all image paths for the current label
        image_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
        
        # Log if there are no images in the class folder
        if len(image_paths) == 0:
            print(f"[WARNING] No images found in {label} class.")
            continue
        
        # Log the number of images found before sampling
        print(f"[INFO] Found {len(image_paths)} images for class '{label}'")
        
        # Reduce the number of images by the given reduction factor
        num_images_to_select = int(len(image_paths) * reduction_factor)
        
        # If there are fewer images than the desired reduction, use all images
        if num_images_to_select == 0:
            print(f"[WARNING] Reduction factor too high, using all {len(image_paths)} images for '{label}'")
            num_images_to_select = len(image_paths)
        
        # Randomly sample the reduced number of images
        sampled_images = random.sample(image_paths, k=num_images_to_select)
        
        # Log how many images were selected
        print(f"[INFO] {len(sampled_images)} images selected for class '{label}'")
        
        # Copy the sampled images to the output directory
        for image_path in sampled_images:
            try:
                shutil.copy(image_path, output_label_dir)
                print(f"[INFO] Copied {image_path} to {output_label_dir}")
            except Exception as e:
                print(f"[ERROR] Error copying {image_path} to {output_label_dir}: {e}")
        
        print(f"[INFO] Class '{label}' has been reduced and copied.")

# Function to directly copy the test dataset without any reduction
def copy_test_dataset(input_dir, output_dir):
    """
    Function to copy all images from the test dataset to the output directory without any reduction.
    
    Args:
    - input_dir: Directory containing the input data.
    - output_dir: Directory to store the unbalanced data.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each class for the test dataset
    for label in class_labels:
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        
        # Make sure the output directory for each label exists
        os.makedirs(output_label_dir, exist_ok=True)
        
        # Check if the label directory exists
        if not os.path.exists(label_dir):
            print(f"[ERROR] Directory for label '{label}' does not exist: {label_dir}")
            continue
        
        # Get all image paths for the current label
        image_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
        
        # Log if there are no images in the class folder
        if len(image_paths) == 0:
            print(f"[WARNING] No images found in {label} class.")
            continue
        
        # Log the number of images being copied for the class
        print(f"[INFO] Copying {len(image_paths)} images from class '{label}' to {output_label_dir}")
        
        # Copy all images to the output directory (no reduction)
        for image_path in image_paths:
            try:
                shutil.copy(image_path, output_label_dir)
                print(f"[INFO] Copied {image_path} to {output_label_dir}")
            except Exception as e:
                print(f"[ERROR] Error copying {image_path} to {output_label_dir}: {e}")
        
        print(f"[INFO] All images from class '{label}' have been copied.")

# Unbalance the training dataset by reducing the number of images
unbalance_train_dataset(train_dir, balanced_train_dir, reduction_factor=0.5)

# Directly copy all images from the test dataset without any reduction
copy_test_dataset(test_dir, balanced_test_dir)

print("Training dataset has been unbalanced and copied. Test dataset has been copied without reduction.")
