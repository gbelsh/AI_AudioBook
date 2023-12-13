import os
import random
import shutil

def split_dataset(source_directory, output_directory, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15):
    assert train_ratio + validation_ratio + test_ratio == 1

    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Create subdirectories for train, validation, and test sets
    train_dir = os.path.join(output_directory, 'train')
    validation_dir = os.path.join(output_directory, 'validation')
    test_dir = os.path.join(output_directory, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]
    random.shuffle(all_files)

    total_files = len(all_files)
    train_size = int(total_files * train_ratio)
    validation_size = int(total_files * validation_ratio)

    for i, file in enumerate(all_files):
        if i < train_size:
            target_dir = train_dir
        elif i < train_size + validation_size:
            target_dir = validation_dir
        else:
            target_dir = test_dir

        shutil.copy2(os.path.join(source_directory, file), os.path.join(target_dir, file))
        txt_file = file.replace('.jpg', '.txt')
        shutil.copy2(os.path.join(source_directory, txt_file), os.path.join(target_dir, txt_file))

    print("Dataset split into training, validation, and test sets.")

if __name__ == '__main__':
    source_directory = 'CC12M'  # Replace with your CC12M folder path
    output_directory = 'Datasets'  # The main directory where the datasets will be stored

    split_dataset(source_directory, output_directory)
