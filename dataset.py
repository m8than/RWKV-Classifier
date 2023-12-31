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
        label_value = 1.0 #28 / len(self.labels[idx])
        label_output = [0.0] * 28
        for id in self.labels[idx]:
            label_output[id] = label_value
        
        x = torch.tensor(tokenized_text)
        y = torch.tensor(label_output)
        
        return x, y