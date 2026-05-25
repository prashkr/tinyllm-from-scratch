import torch
import torch.nn as nn
from dataset import Dataset
from tokenizer import CharTokenizer
from transformer import Transformer
from constants import D_HEAD, D_MODEL, NUM_BLOCKS, SEQUENCE_LENGTH


def prompt(text: str, model: Transformer, tokenizer: CharTokenizer, num_tokens: int = 20, temperature: float= 0.5):
    model.eval()
    prompt_enc = tokenizer.encode(text) # (T)
    x = torch.tensor([prompt_enc]) # (B=1, T)

    softmax = nn.Softmax(dim=-1)
    out_tokens = []
    with torch.no_grad():
        for _ in range(num_tokens): 
            logits = model(x) # (B, T, V)
            last_logits = logits[:, -1, :] # (B, V)
            scaled_logits = last_logits / temperature # (B, V)
            probs = softmax(scaled_logits) # (B, V)
            token = torch.multinomial(probs, num_samples=1) # (B, 1)
            out_tokens.append(int(token[0][0]))
            x = torch.concat((x, token), dim=-1)
            x = x[:, -SEQUENCE_LENGTH:]
    generated_text = text + tokenizer.decode(out_tokens)
    return generated_text


if __name__ == '__main__':
    dataset = Dataset(path='data/input.txt')
    dataset.parse()

    tokenizer = CharTokenizer()
    tokenizer.build(dataset)

    # todo: load model from a checkpoint
    model = Transformer(
        seq_length=SEQUENCE_LENGTH,
        d_model=D_MODEL,
        d_head=D_HEAD,
        vocab_size=tokenizer.vocab_size,
        num_blocks=NUM_BLOCKS
    )
    
    print(prompt('Shakespeare advices everyone to ', model=model, tokenizer=tokenizer, num_tokens=20, temperature=0.5))
