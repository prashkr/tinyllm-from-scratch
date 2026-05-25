# tinyllm

A small from-scratch character-level language model experiment built with PyTorch.

The goal of this project is to understand the moving parts of a GPT-style model by implementing the data loader, character tokenizer, masked self-attention, transformer block, training loop, and sampling loop directly.

## Project Layout

- `dataset.py` - loads `data/input.txt` and samples next-character training windows.
- `tokenizer.py` - builds a character vocabulary from the dataset and encodes/decodes text.
- `transformer.py` - defines masked self-attention, MLP, transformer block, and model head.
- `train.py` - trains the model on random text windows and prints loss/sample generations.
- `generate.py` - samples new characters from a model using temperature sampling.
- `constants.py` - shared training and model configuration.
- `data/input.txt` - Shakespeare-style training corpus.

## Model

Current architecture:

- Character-level tokenizer
- Token embedding
- Causal masked self-attention
- Pre-norm transformer block
- MLP feed-forward block
- Final layer norm
- Linear language-model head

The model predicts the next character at every position:

```text
x:      (B, T)
logits: (B, T, V)
target: (B, T)
```

where:

- `B` is batch size
- `T` is sequence length
- `V` is vocabulary size

## Training

Install dependencies with `uv`, then run:

```bash
uv run python train.py
```

Or, if the virtual environment is already active:

```bash
python train.py
```

The training loop:

1. Samples random contiguous text windows.
2. Encodes input and target text into token IDs.
3. Runs the model to produce logits.
4. Flattens `(B, T, V)` logits to `(B*T, V)`.
5. Computes cross-entropy against flattened targets.
6. Runs backpropagation with AdamW.

A random model starts around:

```text
loss ~= log(vocab_size)
```

For a vocabulary around 65 characters, that is roughly `4.17`. A healthy run should move downward from there, though individual printed losses will jump because each step samples a fresh random batch.

## Generation

`generate.py` uses temperature sampling:

```text
last logits -> divide by temperature -> softmax -> multinomial sample
```

Lower temperature makes output more conservative. Higher temperature makes output more random.

Note: `generate.py` currently creates a fresh model unless trained weights are loaded or generation is called from an active training run. A fresh model will produce random text.

## Current Limitations

- Positional embeddings are still TODO.
- Attention is currently single-head.
- No checkpoint save/load yet.
- No validation split or averaged evaluation loss yet.
- The causal mask is created during forward pass and should later be made device-safe for GPU/MPS.

## Useful Next Steps

1. Add positional embeddings.
2. Save and load model checkpoints.
3. Add validation loss.
4. Add moving-average loss logging.
5. Extend attention to multi-head attention.
