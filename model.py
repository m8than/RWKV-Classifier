import math
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, float32
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import wandb
from modules.SeparatedLinearGate import SeparatedLinearGate
from modules.SeparatedLinear import SeparatedLinear
from sophia import SophiaG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modules.RWKV5Block import RWKV5Block

class RWKV5Classifier(pl.LightningModule):
    def __init__(self, vocab_size, n_embd=256, n_layer=6, head_count=8, dim_ffn=256, dim_att=256, output_dim=6):
        super(RWKV5Classifier, self).__init__()
        
        self.automatic_optimization = False
        
        self.emb = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([RWKV5Block(n_embd, head_count, dim_att, dim_ffn, n_layer, i) for i in range(n_layer)])
        
        self.ln_out = nn.LayerNorm(n_embd)
        
        self.head = nn.Linear(n_embd, output_dim, bias=False)
    
    def forward(self, idx):
        B, T = idx.size()

        x = self.emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)

        x = self.head(x)
        
        # output last
        # x = x[:, -1, :]
        x = F.softmax(x, dim=-1)

        return x
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        self.train()
        
        x, y = batch
        
        yhat = self(x)
        
        opt.zero_grad()
        
        loss = RWKV5Classifier.loss_fn(yhat, y)
        
        self.manual_backward(loss)
        
        #         # Calculate gradient norms before clipping
        # pre_clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))

        # # Gradient Clipping
        # max_grad_norm = 0.5  # Set the maximum gradient norm value
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        # # Calculate gradient norms after clipping
        # post_clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'))

        opt.step()
            
        # calculate accuracy
        acc = (torch.argmax(yhat[:,-1], dim=-1) == torch.argmax(y, dim=-1)).float().mean()
        
        
        self.log('train_loss', loss,  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('pre_clip_norm', pre_clip_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # self.log('post_clip_norm', post_clip_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # wandb.log({"train_loss": loss,
        #              "diversity_diff": diversity_diff,
        #                 "accuracy": acc})
    
    def configure_optimizers(self):
        # adam
        return Adam(self.parameters(), lr=1e-4, betas=(0.965, 0.99), weight_decay=1e-1)
        
        #return SophiaG(self.parameters(), lr=1e-4, betas=(0.965, 0.99), rho=0.1, weight_decay=1e-1)
    
    @staticmethod
    def loss_fn(outputs, labels):
        labels = labels.float()
        # labels = labels.reshape(40, 1, -1)
        # labels = labels.repeat(1, outputs.shape[1], 1)
        
        # get last output
        last_output = outputs[:, -1, :]
        
        # Compute MSE loss
        loss = F.mse_loss(last_output.reshape(-1), labels.reshape(-1))
        return loss
