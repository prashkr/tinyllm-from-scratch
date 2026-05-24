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
import random
import math

BATCH_SIZE = 20 
BLOCK_SIZE = 100 
EMBEDDING_SIZE = 32 


class Dataset:
    def __init__(self, path='data/input.txt') -> None:
        self.path = path
        self.chars = []
    
    def parse(self):
        # return list of chars from the dataset
        lines = open(self.path).readlines()
        for line in lines:
            for c in line:
                self.chars.append(c)


class DatasetLoader:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
    
    def sample(self, batch_size: int = BATCH_SIZE, block_size: int = BLOCK_SIZE, stride: int = 1, random_offset: bool = False):
        samples = []
        start_index = 0
        for i in range(batch_size):
            if random_offset:
                start_index = random.randint(0, len(self.dataset.chars))
            if start_index + block_size + 1 > len(self.dataset.chars):
                continue
            train_slice = self.dataset.chars[start_index : start_index+block_size]
            test_slice = self.dataset.chars[start_index+1 : start_index+block_size+1]
            samples.append((train_slice, test_slice))
            train_s = "".join(train_slice)
            test_s = "".join(test_slice)
            print(f'batch={i+1}\ntrain: {repr(train_s)}\ntest:  {repr(test_s)}\n')
            if not random_offset:
                start_index += stride 
        return samples


class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
    
    def build(self, dataset: Dataset):
        sorted_chars = sorted(set(dataset.chars))
        for i, c in enumerate(sorted_chars):
            self.stoi[c] = i
            self.itos[i] = c 

    def encode(self, seq):
        # convert seq to list of ids 
        output = []
        for c in seq:
            output.append(self.stoi[c])
        return output

    def decode(self, ids):
        # convert ids to seq 
        output = ''
        for _id in ids:
            output += self.itos[_id]
        return output
    
    @property
    def vocab_size(self):
        return len(self.stoi)


######### Transformer ########


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


class TransfomerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x):
        # x dim = (B, T, C)
        pass


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x dim = (B, T, C)
        pass


class Transformer(nn.Module):
    def __init__(self, 
        batch_size: int, 
        seq_length: int, 
        d_model: int,
        vocab_size: int,
        num_heads: int,
        num_transformer_blocks: int
    ) -> None:
        super().__init()
        # pos embedding (should this be part of the transfomer arch?)
        self.B, self.T, self.C = batch_size, seq_length, d_model 
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
    
    def forward(self, x):
        """
        Note: x has a shape of (B, T, C)

        x = token_embedding(x) + pos_embedding(x)
        x = x + attention(layernorm(x))
        x = x + mlp(layernorm(x))
        """
        # token embedding
        x = nn.Embedding(self.vocab_size, self.C)(x)

        # layer norm 
        x_norm = nn.LayerNorm(self.C)(x)

        # attn block
        x_attn = AttentionBlock()(x_norm)
        x = x + x_attn

        x_norm = nn.LayerNorm(self.C)(x) 
        x_mlp = nn.Linear()

