import torch
from dataset import TokenLabelDataset
import os, json
import pandas as pd
import numpy as np
import tokenizers

torch.set_float32_matmul_precision('high')

cur_dir = os.path.dirname(os.path.realpath(__file__))

tokenizer = tokenizers.Tokenizer.from_file(cur_dir + "/20B_tokenizer.json")

input_output_pairs = []

with open(cur_dir + '/setfit_emotion.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        input_output_pairs.append((json_obj['text'], json_obj['label']))
        
# 6 labels
# 0: sadness
# 1: joy
# 2: love
# 3: anger
# 4: fear
# 5: surprise

# tokenize text and output columns
def generate_df(input_output_pairs):
    df = pd.DataFrame(input_output_pairs, columns=['text', 'output'])
    return df

df = generate_df(input_output_pairs)

print(df.head())

# test print random row untokenized
print(df['text'].iloc[0])
print(df['output'].iloc[0])

# print 187 label 
print(df.head())

from dataset import TokenLabelDataset
from model import RWKV5Classifier
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn, save, load, stack

# def collate_fn(batch):
#     # Separate the inputs (batches of tensors)
#     x, y = zip(*batch)

#     # Pad the first input to the maximum length
#     max_length = max([len(seq) for seq in x])
#     x_pad = []
#     for seq in x:
#         padded_seq = nn.functional.pad(seq, pad=(0, max_length - len(seq)), mode='constant', value=0)
#         x_pad.append(padded_seq)

#     # Convert the padded inputs to tensors
#     x = stack(x_pad)
#     y = stack(y)

#     return x, y

# config
epochs = 1
batch_size = 10

def pad_zip(*sequences):
    sequences = list(sequences)
    new_sequences = [torch.empty(0)] * len(sequences[0])
    for j in range(len(sequences)):
        batch = sequences[j]
        for i in range(len(batch)):
            max_length = max(int(bch[i].size(0)) for bch in sequences)
            current_length = int(batch[i].size(0))
            new_tensor = torch.cat((batch[i], torch.zeros(max_length - current_length, dtype=batch[i].dtype))).long()
            new_sequences[i] = torch.cat((new_sequences[i], new_tensor.unsqueeze(0))).long()
            
    return new_sequences


def my_collate_fn(batch):
    return pad_zip(*batch)

dataset = TokenLabelDataset(df['text'].values.tolist(), df['output'].values.tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=True, num_workers=16)

# initti
model = RWKV5Classifier(tokenizer.get_vocab_size(), n_embd=256, n_layer=6, head_count=8, dim_ffn=256, dim_att=256, output_dim=6)
#model.load_state_dict(torch.load(cur_dir + '/model.pt', map_location=torch.device('cuda')))
# opt = SophiaG(model.parameters(), lr=5e-5, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)

loss_fn = nn.CrossEntropyLoss()
 
from tqdm import tqdm
import pytorch_lightning as pl
import wandb
import deepspeed
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# Training flow
if __name__ == '__main__':
    pl.seed_everything(42)  # Set a fixed seed for reproducibility
    
    # wandb.init(project="output-shaper")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',  # Specify the metric to monitor
        dirpath=cur_dir + '/models',  # Specify the directory to save the models
        filename='model-{epoch:02d}-{val_loss:.2f}',  # Specify the filename pattern
        save_top_k=1,  # Save the best model based on the monitored metric
        mode='min'  # Specify the direction of improvement for the monitored metric
    )

    wandb_logger = WandbLogger(project="rwkv5-classifier")
    trainer = pl.Trainer(
        precision='bf16-mixed',
        max_epochs=epochs,
        #strategy="deepspeed_stage_3",
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=dataloader)
    
    # save the lightning model
    

    # Save the model
    torch.save(model.state_dict(), "model.pt")

    wandb.finish()