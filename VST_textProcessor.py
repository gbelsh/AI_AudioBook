import os
from transformers import BertTokenizer, BertModel
import torch

class TextProcessor:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length

    def process_text_from_file(self, text_file):
        # Read text from the file
        with open(text_file, 'r', encoding='utf-8') as file:
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
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get embeddings from BERT
        return self.get_embeddings(input_ids, attention_mask)

    def get_embeddings(self, input_ids, attention_mask):
        # Get embeddings from BERT
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        # Mean pooling of embeddings
        embeddings = embeddings.mean(dim=1)
        return embeddings
