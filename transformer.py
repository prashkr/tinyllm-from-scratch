# dataset init
# dataset loader
    # context size
    # stride
    # how to handle start of dataset and end of dataset
# for sample in dataset: 
    # train, test split
    # tokenizer
# token embedding: list of token ids 0 -> list of token embedding vectors
# positional embedding: list of vectors -> modified list of vectors
# attention block
# MLP block
# RMSNorm or LayerNorm block
# Attention head
# train
# backprop
# see results 

# You are all resolved rather to die than to famish?
# train -> test
# y -> o
# y o -> o u
# y o u -> o u <space>
# y o u <space> -> o u <space> a
# y o u <space> a -> o u <space> a r

import torch
import torch.nn as nn
import math


"""
Transformer(
  (token_embedding): Embedding(65, 64)
  (blocks): ModuleList(
    (0-1): 2 x TransformerBlock(
      (attention): MaskedSelfAttention(
        (Q): Linear(in_features=64, out_features=16, bias=True)
        (K): Linear(in_features=64, out_features=16, bias=True)
        (V): Linear(in_features=64, out_features=16, bias=True)
        (softmax): Softmax(dim=-1)
        (output_projection): Linear(in_features=16, out_features=64, bias=True)
      )
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True, bias=True)
      (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True, bias=True)
      (mlp): MLP(
        (linear): Linear(in_features=64, out_features=256, bias=True)
        (activation_fn): GELU(approximate='none')
        (output_projection): Linear(in_features=256, out_features=64, bias=True)
      )
    )
  )
  (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True, bias=True)
  (head): Linear(in_features=64, out_features=65, bias=True)
)
"""


class MaskedSelfAttention(nn.Module):
    """
    attention score = softmax(qkT/root(d_head) + mask) @ V
    """

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_model, self.d_head = d_model, d_head
        self.Q = nn.Linear(d_model, d_head) # (B, T, C) @ (C, H) = (B, T, H)
        self.K = nn.Linear(d_model, d_head) # (B, T, C) @ (C, H) = (B, T, H)
        self.V = nn.Linear(d_model, d_head) # (B, T, C) @ (C, H) = (B, T, H)
        self.softmax = nn.Softmax(dim=-1) # softmax across last dim 
        self.output_projection = nn.Linear(d_head, d_model) # final conversion to (B, T, C) shape

    def forward(self, x):
        # x dim = (B, T, C)
        _, T, _ = x.shape
        q = self.Q(x) # (B, T, H)
        k = self.K(x) # (B, T, H)
        v = self.V(x) # (B, T, H)

        qkt = q @ torch.transpose(k, -2, -1) # (B, T, H) @ (B, H, T) = (B, T, T)
        scaled_qkt = qkt / math.sqrt(self.d_head)

        mask = torch.tril(torch.ones(T, T)) # lower triangular matrix with all 1s and 0s
        masked_qkt = scaled_qkt.masked_fill(mask == 0, float("-inf")) # transform to -inf

        attn_weights = self.softmax(masked_qkt) # (B, T, T)
        attn = attn_weights @ v # (B, T, T) @ (B, T, H) = (B, T, H)
        return self.output_projection(attn) 


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_mlp)
        self.activation_fn = nn.GELU()
        self.output_projection = nn.Linear(d_mlp, d_model)
    
    def forward(self, x):
        expanded = self.linear(x)
        activated = self.activation_fn(expanded)
        projected = self.output_projection(activated)
        return projected


class TransformerBlock(nn.Module):
    """
    x = x + attn(norm(x))
    x = x + mlp(norm(x))
    """
    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.attention = MaskedSelfAttention(d_model, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4 * d_model)

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_out = self.attention(attn_input)
        x = x + attn_out

        mlp_input = self.norm2(x)
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out
        return x


class Transformer(nn.Module):
    def __init__(self, 
        seq_length: int, 
        d_model: int,
        d_head: int,
        vocab_size: int,
        num_blocks: int
    ) -> None:
        super().__init__()
        # pos embedding (should this be part of the transfomer arch?)
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks 

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_head) 
            for _ in range(num_blocks)
        ]) 
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Note: x has a shape of (B, T)
        """
        # token embedding
        x = self.token_embedding(x) # (B, T) -> (B, T, C) via a lookup table 

        # todo: pos embedding

        for block in self.blocks:
            x = block(x)
        x = self.norm(x) 
        return self.head(x) # (B, T, C) -> (B, T, V)
         