# Transformer from Scratch: English → French Translation

This repo has my PyTorch implementation of the Encoder–Decoder Transformer from the 2017 paper **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**. I trained it on English-to-French machine translation.

I built it to learn how the architecture works, so it is a smaller version of the original model (2 layers instead of 6). Everything is in one notebook: `encoder_decoder.ipynb`.

## What is in the repo

1. Tokenization and vocabulary (word level, built from scratch)
2. Sinusoidal positional encoding
3. Two versions of multi-head attention (see below)
4. Feed-forward network, encoder, decoder, and the full Transformer
5. Training loop with early stopping
6. Greedy decoding (one sentence, and batched) and BLEU evaluation
7. Export of `config.json`, tokenizer files, and `model.safetensors`

## Architecture

The model uses three attention modules. I wrote them in two different styles on purpose, to understand both:

| Module | Where | Style |
|-|-|-|
| Self-attention | Encoder | **Explicit parameters:** a separate `Wq`, `Wk`, `Wv` matrix for each head, and `einsum` to apply them |
| Masked self-attention | Decoder | **Merged linear:** one `Linear(d_model, d_model)` each for Q, K, V, then the result is split into heads |
| Cross-attention | Decoder | **Merged linear:** same as above; queries come from the decoder, keys and values from the encoder |

**What is the same as the paper:** sinusoidal positional encoding, scaled dot-product attention, 8 heads, `d_model = 512`, feed-forward size `4 × d_model` (2048) with ReLU, residual connections, post-norm layout, causal mask in the decoder, teacher forcing during training.

**What is different from the paper:**

- 2 encoder and 2 decoder layers (paper: 6)
- `RMSNorm` instead of `LayerNorm`
- Word-level vocabulary instead of BPE
- No embedding scaling by `sqrt(d_model)`, and no weight sharing between embeddings and the output layer
- `AdamW` with a constant learning rate, no warmup
- Dropout `0` on attention and residual paths, `0.2` only in the feed-forward block (paper: `0.1` everywhere)
- **No padding mask** (see Known Issues below)

## Dataset

I used the English–French sentence-pair dataset from Kaggle (`eng-fra.txt`). The notebook downloads it automatically.

**Preprocessing:** lowercase, split on spaces, and strip punctuation from the start and end of each word. Each English sentence ends with `<eos>`. Each French sentence starts with `<bos>` and ends with `<eos>`.

**Split:** 85% train, 15% test (random, `random_state=42`).

| | English | French |
|-|-|-|
| Vocabulary size (with special tokens) | 17,598 | 34,222 |
| Longest sentence (in tokens) | 56 | 68 |

## Model Config and Hyperparameters

The model has **58,772,910 trainable parameters**. In `float32`, that is about **224.2 MiB**.

| Parameter | Value |
|-|-|
| Encoder vocab size | 17,598 |
| Decoder vocab size | 34,222 |
| Model dimension | 512 |
| Encoder max sequence length | 56 |
| Decoder max sequence length | 68 |
| Attention heads | 8 |
| Layers (encoder and decoder) | 2 |
| Attention dropout | 0 |
| Residual dropout | 0 |
| Feed-forward dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 228 |
| Loss | Cross-entropy (padding ignored) |
| Max epochs | 20 |

Training stops as soon as the validation loss does not improve for one epoch. The best checkpoint is saved.

## Training Results

I set 20 epochs as the maximum, but the model started to overfit after about 5 epochs, so early stopping ended the run.

- Best train loss: `0.81`
- Best validation loss: `1.28`

## Evaluation

Translation is greedy decoding (always pick the most likely next token). BLEU is computed with `sacrebleu`.

| Method | BLEU |
|-|-|
| One sentence at a time | 25.58 |
| Batched (current code) | 2.65 |

The batched number is **not reliable**. It is caused by a bug in my batched decoding code (see below), so please use the single-sentence number for now.

Note: the same 15% split is used for early stopping, checkpoint selection, and BLEU. There is no separate test set, so the scores are probably a bit optimistic.

## Known Issues

**1. Batched BLEU is wrong because of a bug in `batch_index_lookup`.**
When two or more sentences in a batch finish in the same step, the index map is updated one row at a time using old indexes. After the first removal, the map has already shifted, so the second removal deletes the wrong entry. Finished sentences are then written to the wrong positions (some overwrite others, some slots stay empty), and the hypotheses no longer match their references. Example with 4 sentences where rows 0 and 1 finish together: the map becomes `{0: 1, 1: 3}` but it should be `{0: 2, 1: 3}`.

**2. There is no padding mask.**
Every English sentence is padded to length 56, but the average sentence is much shorter, so most encoder positions are padding. The encoder self-attention and the decoder cross-attention still attend to those positions. This can weaken the link between the input and the output. It is my main suspect for this problem: with 0.2 dropout everywhere, the model reached a validation loss of `1.24` and learned French sentence structure, but the translations did not match the input meaning. It also means training (padded input) and single-sentence inference (unpadded input) do not see the same kind of input.


## Files

- `encoder_decoder.ipynb`: full code
- `config.json`: architecture configuration
- `encoder_tokenizer.json`, `decoder_tokenizer.json`: vocabularies
- `model.safetensors`: trained weights

## Feedback

If you find a bug or have an idea to improve this, please open an issue or a pull request.