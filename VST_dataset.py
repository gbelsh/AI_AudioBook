import os
import random
import shutil

def split_dataset(source_directory, destination_directory, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15):
    # List all files in the source directory
    all_files = [f for f in os.listdir(source_directory) if f.endswith('.txt')]
    
    # Shuffle the files to ensure random distribution
    random.shuffle(all_files)

    # Calculate the number of files for each set
    total_files = len(all_files)
    train_size = int(total_files * train_ratio)
    validation_size = int(total_files * validation_ratio)
    
    # Split files into training, validation, and testing
    train_files = all_files[:train_size]
    validation_files = all_files[train_size:train_size + validation_size]
    test_files = all_files[train_size + validation_size:]

    # Function to copy files to a target directory
    def copy_files(files, target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for f in files:
            shutil.copy2(os.path.join(source_directory, f), os.path.join(target_dir, f))

    # Copy files to respective directories
    copy_files(train_files, os.path.join(destination_directory, 'train'))
    copy_files(validation_files, os.path.join(destination_directory, 'validation'))
    copy_files(test_files, os.path.join(destination_directory, 'test'))

    print("Dataset split into training, validation, and test sets.")

# Example usage
source_directory = 'Project_Gutenberg_Data'  # Directory where the downloaded books are stored
destination_directory = 'Datasets'     # Directory where the dataset will be organized

split_dataset(source_directory, destination_directory)
