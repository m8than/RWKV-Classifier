import torch, os
from model import RWKV5Classifier
import tokenizers
import numpy as np

cur_dir = os.path.dirname(os.path.realpath(__file__))
tokenizer = tokenizers.Tokenizer.from_file(cur_dir + "/20B_tokenizer.json")
#model = OutputShaper.load_from_checkpoint(cur_dir + '/model.pt', input_size=tokenizer.get_vocab_size(), hidden_size=32, k=64)

# load state dict
model = RWKV5Classifier(tokenizer.get_vocab_size(), n_embd=256, n_layer=6, head_count=8, dim_ffn=256, dim_att=256, output_dim=6)
model.load_state_dict(torch.load(cur_dir + '/model.pt', map_location=torch.device('cpu')))

prompt = "i love that thing man"
tokenized_prompt = tokenizer.encode(prompt).ids

# run inference
output = model(torch.tensor([tokenized_prompt]))

list_out = output.tolist()[0]

# label output
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# list to label dictionary
dict_out = dict(zip(labels, list_out))

print(dict_out)