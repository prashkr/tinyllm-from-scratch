from dataset import Dataset, DatasetLoader
from tokenizer import CharTokenizer
from transformer import Transformer
from constants import BATCH_SIZE, LEARNING_RATE, MAX_STEPS, SEQUENCE_LENGTH, D_HEAD, D_MODEL, NUM_BLOCKS
import torch
import torch.nn as nn


def train():
    # sample train, test batches
    # forward pass through the model
    # compute loss
    # backward pass through the model with optimizer set
    # update params
    dataset = Dataset(path='data/input.txt')
    dataset.parse()

    dataloader = DatasetLoader(dataset) 

    tokenizer = CharTokenizer()
    tokenizer.build(dataset)

    model = Transformer(
        seq_length=SEQUENCE_LENGTH,
        d_model=D_MODEL,
        d_head=D_HEAD,
        vocab_size=tokenizer.vocab_size,
        num_blocks=NUM_BLOCKS
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # training loop
    for step in range(MAX_STEPS):
        samples = dataloader.sample(
            batch_size=BATCH_SIZE,
            seq_length=SEQUENCE_LENGTH,
            random_offset=True
        )
        
        x = []
        y = []
        for sample in samples:
            train, target = sample
            train_enc = tokenizer.encode(train)
            target_enc = tokenizer.encode(target)
            x.append(train_enc)
            y.append(target_enc)

        # needs (B, T) tensor
        x = torch.tensor(x) # (B, T)
        y = torch.tensor(y) # (B, T)

        logits = model(x) # (B, T, V)

        # reshape for cross entropy loss 
        # logits: (N, V)
        # targets: (N)
        B, T, V = logits.shape
        logits_flat = logits.reshape(B*T, V)
        y_flat = y.reshape(B*T)

        loss = loss_fn(logits_flat, y_flat)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == MAX_STEPS - 1:
            print(f"step={step:04d} loss={loss.item():.4f}")


if __name__ == '__main__':
    train() 
