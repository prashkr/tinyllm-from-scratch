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

import random

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
