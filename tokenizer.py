from dataset import Dataset

BATCH_SIZE = 20 
BLOCK_SIZE = 100 
EMBEDDING_SIZE = 32 


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