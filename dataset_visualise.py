import os
import matplotlib.pyplot as plt

def count_images_in_directory(directory):
    # Initialize a dictionary to store the class-wise count
    class_count = {}
    
    # Loop through each subdirectory (class) in the directory
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        
        # Ensure that it's a directory
        if os.path.isdir(class_path):
            # Count the number of image files in the class directory
            image_count = sum([1 for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            class_count[class_name] = image_count
            
    return class_count

def plot_image_counts(train_class_count, val_class_count):
    # Set up the figure for plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot for training data
    axes[0].bar(train_class_count.keys(), train_class_count.values(), color='blue')
    axes[0].set_title('Training Set Class Counts')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Number of Images')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot for validation data
    axes[1].bar(val_class_count.keys(), val_class_count.values(), color='green')
    axes[1].set_title('Validation Set Class Counts')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Number of Images')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Display the plots
    plt.tight_layout()
    plt.show()

def main():
    # Provide the paths to your train and validation directories
    train_directory = '/home/saivarun/AD/final_dataset/train'
    val_directory = '/home/saivarun/AD/final_dataset/test'
    
    # Count images in training and validation directories
    train_class_count = count_images_in_directory(train_directory)
    val_class_count = count_images_in_directory(val_directory)
    
    # Print the counts
    print("Training Set Class Counts:")
    for class_name, count in train_class_count.items():
        print(f"{class_name}: {count} images")
    
    print("\nValidation Set Class Counts:")
    for class_name, count in val_class_count.items():
        print(f"{class_name}: {count} images")
    
    # Visualize the counts using bar charts
    plot_image_counts(train_class_count, val_class_count)

if __name__ == "__main__":
    main()

