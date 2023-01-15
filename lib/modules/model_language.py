import logging
import torch.nn as nn
import torch
from fastai.vision import *
import torch.nn.functional as F

from lib.modules.transformer import (PositionalEncoding, 
                                 TransformerDecoder,
                                 TransformerDecoderLayer)
_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048,  # 1024
                          dropout=0.1, activation='relu')

class BCNLanguage(nn.Module):
    def __init__(self, nclass,max_length):
        super().__init__()
        d_model =  _default_tfmer_cfg['d_model']
        nhead = _default_tfmer_cfg['nhead']
        d_inner = _default_tfmer_cfg['d_inner']
        dropout = _default_tfmer_cfg['dropout']
        activation = _default_tfmer_cfg['activation']
        num_layers = 4
        self.d_model = d_model
        self.detach = True
        self.use_self_attn = False
        self.loss_weight = 1.0
        self.max_length = max_length + 1  # additional stop token
        self.debug = False

        self.proj = nn.Linear(nclass, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, nclass)



    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        # if self.detach: tokens = tokens.detach()
        tokens = torch.exp(tokens)
        tokens = tokens.permute(1, 0, 2)
        # self.max_length,_ = torch.max(lengths,dim=-1,keepdim=False)
        # self.max_length += 1
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = _get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = _get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed)  # (T, N, E)
        #,
                # tgt_key_padding_mask=padding_mask,
                # memory_mask=location_mask,
                # memory_key_padding_mask=padding_mask
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)

        logits = logits.permute(1, 0, 2)
        # pt_lengths = self._get_length(logits)

        # res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
        #         'loss_weight':self.loss_weight, 'name': 'language'}
        return torch.softmax(logits,-1)


def _get_padding_mask(length, max_length):
    length = length.unsqueeze(-1)
    grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
    return grid >= length

    
def _get_location_mask(sz, device=None):

    mask = torch.eye(sz, device=device)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))
    return mask