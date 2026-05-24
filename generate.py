import torch
import torch.nn as nn
from dataset import Dataset, DatasetLoader
from tokenizer import CharTokenizer
from transformer import Transformer
from constants import D_HEAD, D_MODEL, NUM_BLOCKS, SEQUENCE_LENGTH


def generate(model: Transformer, x: torch.Tensor, temperature: float = 0.5):
    """
    x dim = (B, T) 
    for generate B = 1 (local testing)
    """
    with torch.no_grad():
        softmax = nn.Softmax(dim=-1)
        # run model
        logits = model(x) # (B, T, V)

        last_logits = logits[:, -1, :] # (B, V)
        scaled_logits = last_logits / temperature # (B, V)

        # softmax
        probs = softmax(scaled_logits) # (B, V)

        # sample a token ids, one per batch
        return torch.multinomial(probs, num_samples=1) # (B, 1)
        

def prompt(text: str, model: Transformer, tokenizer: CharTokenizer, num_tokens: int = 10):
    prompt_enc = tokenizer.encode(text) # (T)
    x = torch.tensor([prompt_enc]) # (B=1, T)
    out_tokens = []
    for _ in range(num_tokens): 
        tok = generate(model, x)  # (B, 1)
        out_tokens.append(tok)

        x = torch.concat((x, tok), dim=1) # (B, T+1)

        # truncate to max seq length 
        x = x[:, -SEQUENCE_LENGTH:]

    tokenizer.decode(out_tokens.tolist())


if __name__ == '__main__':
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
    model.eval()
    
    prompt('Shakespeare advices everyone to ', model=model, tokenizer=tokenizer)