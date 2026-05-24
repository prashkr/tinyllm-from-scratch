from dataset import Dataset, DatasetLoader
from tokenizer import CharTokenizer

ds = Dataset()
ds.parse()

tok = CharTokenizer()
tok.build(ds)

tok.stoi
tok.itos

tok.decode(tok.encode('Before we proceed any further, hear me speak.'))

loader = DatasetLoader(ds)
batch_size = 10
block_size = 50 
stride = 1 
random_offset = False 
samples = loader.sample(batch_size, block_size, stride, random_offset)
for train_slice, test_slice, in samples:
    train_slice_str = ''.join(train_slice)
    test_slice_str = ''.join(test_slice)
    repr(tok.decode(tok.encode(train_slice_str)))
