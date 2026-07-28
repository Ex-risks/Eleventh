"""A small bidirectional Transformer that outputs a token distribution at every
position. Full self-attention, no causal mask -- this is an encoder, not a
left-to-right generator. Used as the denoiser for masked diffusion."""

import math
import mlx.core as mx
import mlx.nn as nn

import config as C


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C.D_MODEL)
        self.attn = nn.MultiHeadAttention(C.D_MODEL, C.N_HEADS, bias=True)
        self.ln2 = nn.LayerNorm(C.D_MODEL)
        self.fc1 = nn.Linear(C.D_MODEL, C.FFN_DIM)
        self.fc2 = nn.Linear(C.FFN_DIM, C.D_MODEL)
        self.drop = nn.Dropout(C.DROPOUT)

    def __call__(self, x):
        h = self.ln1(x)
        # No mask: every position attends to every other (bidirectional).
        x = x + self.drop(self.attn(h, h, h))
        h = self.ln2(x)
        x = x + self.drop(self.fc2(nn.gelu(self.fc1(h))))
        return x


class DiffusionLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int = C.SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, C.D_MODEL)
        self.pos_emb = nn.Embedding(seq_len, C.D_MODEL)
        self.drop = nn.Dropout(C.DROPOUT)
        self.blocks = [Block() for _ in range(C.N_LAYERS)]
        self.ln_f = nn.LayerNorm(C.D_MODEL)
        self.head = nn.Linear(C.D_MODEL, vocab_size)

    def __call__(self, idx):
        # idx: (B, T) int32
        T = idx.shape[1]
        pos = mx.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, VOCAB)


def n_params(model):
    import mlx.utils
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
