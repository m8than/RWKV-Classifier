import torch
from torch.utils.data import Dataset
from torch import tensor
import numpy as np

class TokenLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokenized_text = self.tokenizer.encode(self.texts[idx]).ids
        label_output = [0.0] * 6
        label_output[self.labels[idx]] = 1.0
        
        x = torch.tensor(tokenized_text)
        y = torch.tensor(label_output)
        
        return x, y