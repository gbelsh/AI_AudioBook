import os
from torch.utils.data import Dataset



class VSTDataset(Dataset):
    def __init__(self, directory, tokenizer, max_length=512):
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.filenames = os.listdir(directory)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Tokenize and encode the text
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return inputs['input_ids'], inputs['attention_mask']

