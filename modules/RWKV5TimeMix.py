import torch.nn as nn
import torch
from torch.nn import functional as F

class RWKV_TimeMix_RWKV5_R2R3(nn.Module):
    def __init__(self, head_count, dim_att, n_layer, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = 64
        self.n_head = dim_att // self.head_size
        self.head_size_divisor = 8

        self.chunk_len = 512

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.gate = nn.Linear(n_embd, dim_att, bias=False)

            # fancy time_decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.key = nn.Linear(n_embd, dim_att, bias=False)
        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, dim_att)
        
        
    def jit_func(self, x):
        B, TT, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
        k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
        v = self.value(xv).view(B, TT, self.n_head, -1).transpose(1, 2)                 # BTC -> BHTS
        g = F.silu(self.gate(xg))

        return r, k, v, g
    
    
    def jit_func_2(self, r, k, v, g, w, wk, wb, ws):
        B, H, TT, S = r.size()
        T = self.chunk_len

        s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
        x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

        for i in range(TT // T):
            rr = r[:, :, i*T:i*T+T, :]
            kk = k[:, :, :, i*T:i*T+T]
            vv = v[:, :, i*T:i*T+T, :]

            x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

            s = ws * s + (kk * wk) @ vv
        
        x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
        x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S) * g
        return self.output(x)
    
    def forward(self, x):
        H = self.n_head
        T = self.chunk_len

        r, k, v, g = self.jit_func(x)

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)
        
        u = self.time_faaaa.float().unsqueeze(-1)
        
        
        ws = w.pow(T).reshape(1, H, 1, 1)

        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)

        wk = w.reshape(1, H, 1, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(1, H, T, T)

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)
        
        return self.jit_func_2(r, k, v, g, w, wk, wb, ws)