import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding

class HarmonIA(nn.Module):
    def __init__(self,ntokens,d_model, n_heads, n_layers, ff_dim, device, padd_index = None,*args, **kwargs,) -> None:
        super(HarmonIA,self).__init__(*args, **kwargs)
        self.device = device
        self.emb = nn.Embedding(ntokens,d_model,device=device, padding_idx= padd_index)
        self.pe = PositionalEncoding(d_model,0.1,1000,device)
        self.layer = nn.TransformerDecoderLayer(d_model,n_heads,dim_feedforward=ff_dim,batch_first=True, device=device)
        self.DecoderLayers = nn.TransformerDecoder(self.layer,n_layers)
        self.linear = nn.Linear(d_model,ntokens, device=device)
    def forward(self,x):
        mask = self.get_tgt_mask(x.shape[1])
        pad_mask = None
        tgt = self.emb(x)
        tgt = self.pe(tgt)
        tgt = tgt.to(self.device)
        out = self.DecoderLayers(tgt,tgt,tgt_mask = mask, memory_mask = mask, tgt_key_padding_mask = pad_mask, memory_key_padding_mask = pad_mask)
        
        return self.linear(out)
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size, device=self.device) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        matrix = matrix.float()
        pad_mask = matrix.masked_fill(matrix != pad_token, float(0.0))
        pad_mask = matrix.masked_fill(pad_mask == pad_token, float(1.0))
        return pad_mask
        