from main import Dataset, DatasetLoader, Tokenizer
ds = Dataset()
ds.parse()

tok = Tokenizer()
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
