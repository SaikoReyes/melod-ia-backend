import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding

class MelodIA(nn.Module):
    def __init__(self,ntokens,d_model, n_heads, ff_dim, device, padd_index = None,*args, **kwargs,) -> None:
        super(MelodIA,self).__init__(*args, **kwargs)
        self.dim_model = d_model
        self.device = device
        self.emb = nn.Embedding(ntokens,d_model,device=device, padding_idx= padd_index)
        self.pe = PositionalEncoding(d_model,0.1,2000,device)
        self.transformer = nn.Transformer(d_model,n_heads, dim_feedforward=ff_dim,batch_first=True, device=device, num_decoder_layers=3, num_encoder_layers=3)
        self.linear = nn.Linear(d_model,ntokens, device=device)
    def forward(self,src,tgt):
        mask = self.get_tgt_mask(tgt.shape[1])
        src_mask = self.get_tgt_mask(src.shape[1])
        pad_mask = None
        tgt = self.emb(tgt)
        src = self.emb(src)
        
        tgt = self.pe(tgt)
        src = self.pe(src)
        
        tgt = tgt.to(self.device)
        out = self.transformer(src,tgt,src_mask = src_mask, tgt_mask = mask)
        
        return self.linear(out)
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size, device=self.device) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask   