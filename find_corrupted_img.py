import PIL
from PIL import Image
import os

def find_and_delete_corrupted_images(directory):
    """
    Finds corrupted image files in the given directory, deletes them, 
    and also deletes their corresponding text files.

    :param directory: Path to the directory containing image files.
    """
    corrupted_files = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it is, in fact, an image
            except (IOError, SyntaxError, PIL.UnidentifiedImageError) as e:
                print(f"Corrupted image file found: {file_path} - Error: {e}")
                corrupted_files.append(file_path)

                # Delete the corrupted image file
                os.remove(file_path)
                print(f"Deleted corrupted image file: {file_path}")

                # Delete the corresponding text file
                text_file_path = os.path.splitext(file_path)[0] + '.txt'
                if os.path.exists(text_file_path):
                    os.remove(text_file_path)
                    print(f"Deleted corresponding text file: {text_file_path}")

    return corrupted_files

if __name__ == '__main__':
    directory = 'Datasets/validation'  # Replace with your directory path
    corrupted_files = find_and_delete_corrupted_images(directory)
    print(f"Total corrupted files: {len(corrupted_files)}")
