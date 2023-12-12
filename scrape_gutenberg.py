import requests
import time
import os

def download_book(book_id, output_directory):
    # Gutenberg URL pattern for plain text UTF-8 books
    url = f'http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(output_directory, f'{book_id}.txt')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(response.text)
            print(f'Book {book_id} downloaded successfully.')
        else:
            print(f'Failed to download book {book_id}. HTTP Status Code: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading book {book_id}: {e}')

def main():
    # List of Project Gutenberg book IDs
    book_ids = [
    84,    # "Frankenstein" by Mary Shelley
    1342,  # "Pride and Prejudice" by Jane Austen
    1080,  # "A Modest Proposal" by Jonathan Swift
    1661,  # "The Adventures of Sherlock Holmes" by Arthur Conan Doyle
    2701,  # "Moby Dick" by Herman Melville
    2600,  # "War and Peace" by Leo Tolstoy
    1232,  # "The Prince" by Niccolò Machiavelli
    174,   # "The Picture of Dorian Gray" by Oscar Wilde
    345,   # "Dracula" by Bram Stoker
    6130,  # "Metamorphosis" by Franz Kafka
    219,   # "Heart of Darkness" by Joseph Conrad
    76,    # "Adventures of Huckleberry Finn" by Mark Twain
    5200,  # "Ulysses" by James Joyce
    98,    # "A Tale of Two Cities" by Charles Dickens
    43,    # "The Strange Case of Dr. Jekyll and Mr. Hyde" by Robert Louis Stevenson
    205,   # "Walden" by Henry David Thoreau
    160,   # "The Awakening, and Selected Short Stories" by Kate Chopin
    2814,  # "Dubliners" by James Joyce
    1952,  # "The Yellow Wallpaper" by Charlotte Perkins Gilman
    3207,  # "Leaves of Grass" by Walt Whitman
    ]

    output_directory = 'Project_Gutenberg_Data'  # Set your desired output directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for book_id in book_ids:
        download_book(book_id, output_directory)
        time.sleep(60)  # 60-second delay between downloads to be respectful of server load

if __name__ == '__main__':
    main()
