import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class VSTDataset(Dataset):
    def __init__(self, directory, tokenizer, max_length=512, transform=None):
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or transforms.ToTensor()  # Default transform
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.txt')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        txt_filename = os.path.join(self.directory, self.filenames[idx])
        img_filename = txt_filename.replace('.txt', '.jpg')  # Assuming image file is .jpg

        # Load and process the image
        image = Image.open(img_filename).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load and tokenize the text
        with open(txt_filename, 'r', encoding='utf-8') as file:
            text = file.read()
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), image

# Example usage
# tokenizer =
