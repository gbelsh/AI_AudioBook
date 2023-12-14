import csv
import os
import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError

def download_image(image_url, output_directory, file_name):
    try:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()  # This will raise an HTTPError for bad responses
        with open(os.path.join(output_directory, file_name), 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return True
    except HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except Timeout:
        print("The request timed out")
    except ConnectionError:
        print("Connection error occurred")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False

def load_cc12m_dataset(tsv_file, output_directory, max_items=None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(tsv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')

        for i, row in enumerate(reader):
            if max_items and i >= max_items:
                break

            image_url, description = row
            image_file_name = f'image_{i}.jpg'
            success = download_image(image_url, output_directory, image_file_name)

            if success:
                # Save the description
                text_file_name = os.path.join(output_directory, f'image_{i}.txt')
                with open(text_file_name, 'w', encoding='utf-8') as text_file:
                    text_file.write(description)

                print(f'Downloaded and saved image {i}')

if __name__ == '__main__':
    tsv_file = 'C:/Users/zerod/Downloads/cc12m.tsv'  # Path to your CC12M tsv file
    output_directory = 'CC12M'  # Directory where images and descriptions will be saved

    load_cc12m_dataset(tsv_file, output_directory, max_items=10000)  # Set max_items as needed
