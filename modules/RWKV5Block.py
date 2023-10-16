import torch.nn as nn
import torch
from .RWKV5TimeMix import RWKV_TimeMix_RWKV5_R2R3
from .RWKVChannelMix import RWKV_ChannelMix

class RWKV5Block(nn.Module):
    def __init__(self, n_embd, head_count, dim_att, dim_ffn, n_layer, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix_RWKV5_R2R3(head_count, dim_att, n_layer, n_embd, layer_id)
        self.ffn = RWKV_ChannelMix(n_embd, n_layer, dim_ffn, layer_id)
        
    def forward(self, x, x_emb=None):
        B, T, C = x.size()
        
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
