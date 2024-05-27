import pandas as pd
import torch
from torch.utils.data import Dataset

#TODO: rewrite for sql
class RakutenTextDataset(Dataset):
    def __init__(self, file_path, text_column, label_column, vocab, spacy_model):
        self.vocab = vocab
        self.text_data, self.labels = self.load_data(file_path, text_column, label_column)
        self.tokenizer =  spacy_model.tokenizer

    def load_data(self, file_path, text_column, label_column):
        # Load the CSV file
        df = pd.read_csv(file_path)
        text_data = df[text_column].tolist()
        labels = df[label_column].tolist()
        return text_data, labels
    
    def text_to_tensor(self, text):
        tokens = self.tokenizer(text)
        tensor = [self.vocab[token.text] for token in tokens if token.text in self.vocab]
        return torch.tensor(tensor, dtype=torch.long) 

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, index):
        text = self.text_data[index]
        label = self.labels[index]
        text_tensor = self.text_to_tensor(text)
        return text_tensor, label
