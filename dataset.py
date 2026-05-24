import random


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
    
    def sample(self, batch_size: int, seq_length: int, stride: int = 1, random_offset: bool = False, debug=False):
        samples = []
        start_index = 0
        for i in range(batch_size):
            if random_offset:
                start_index = random.randint(0, len(self.dataset.chars))
            if start_index + seq_length + 1 > len(self.dataset.chars):
                continue
            train_slice = self.dataset.chars[start_index : start_index+seq_length]
            test_slice = self.dataset.chars[start_index+1 : start_index+seq_length+1]
            samples.append((''.join(train_slice), ''.join(test_slice)))
            train_s = "".join(train_slice)
            test_s = "".join(test_slice)
            if debug:
                print(f'batch={i+1}\ntrain: {repr(train_s)}\ntest:  {repr(test_s)}\n')
            if not random_offset:
                start_index += stride 
        return samples