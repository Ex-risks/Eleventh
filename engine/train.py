"""Masked-diffusion training on a single text corpus, over a BPE vocabulary
learned from that same corpus.

Forward process: independently mask each token with probability t, where t is
sampled per example from U(0.1, 1]. The model recovers original tokens at masked
positions. Loss is CE over masked positions with 1/t reweighting (LLaDA/MDLM).
"""

import math
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import config as C
from model import DiffusionLM, n_params
from tokenizer import train_bpe


def load_corpus(path: str):
    """Read the file, learn a BPE vocabulary from it, and return (tokens, tokenizer)."""
    with open(path, "rb") as f:
        raw = f.read()
    # If the file is larger than the budget, sample a contiguous window randomly so
    # every run is a slightly different slice — keeps endless generation fresh.
    if len(raw) > C.MAX_TRAIN_BYTES:
        start = random.randint(0, len(raw) - C.MAX_TRAIN_BYTES)
        raw = raw[start : start + C.MAX_TRAIN_BYTES]
    text = raw.decode("utf-8", errors="replace")

    tokenizer = train_bpe(text, C.BPE_VOCAB_SIZE)
    tokens = np.array(tokenizer.encode(text), dtype=np.int32)
    return tokens, tokenizer


def make_batches(tokens: np.ndarray, batch_size: int, seq_len: int):
    """Yield infinite (x,) batches of shape (B, T) from the flat token array."""
    n = len(tokens)
    while True:
        idxs = np.random.randint(0, n - seq_len, size=batch_size)
        batch = np.stack([tokens[i : i + seq_len] for i in idxs])
        yield mx.array(batch)


def mask_batch(x: mx.array, mask_id: int):
    """Apply random independent masking with a per-example mask ratio
    t ~ U(0.1, 1.0) so every batch mixes noise levels.
    Returns (noised_x, mask_bool, t) where mask_bool is True at masked
    positions and t has shape (B, 1).
    """
    B, T = x.shape
    t = np.random.uniform(0.1, 1.0, size=(B, 1))
    mask = mx.array(np.random.rand(B, T) < t)
    noised = mx.where(mask, mx.full(x.shape, mask_id, dtype=x.dtype), x)
    return noised, mask, mx.array(t.astype(np.float32))


def loss_fn(model, x, mask_id):
    noised, mask, t = mask_batch(x, mask_id)
    logits = model(noised)                        # (B, T, V)
    # Cross-entropy at masked positions only.
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    x_flat = x.reshape(B * T)
    mask_flat = mask.reshape(B * T)
    log_probs = nn.log_softmax(logits_flat, axis=-1)
    target_log = log_probs[mx.arange(B * T), x_flat]
    masked_loss = mx.where(mask_flat, -target_log, mx.zeros_like(target_log))
    # MDLM/LLaDA objective: weight each masked position by 1/t and normalize by
    # ALL positions, not just the masked ones. Since ~t*T positions are masked,
    # the 1/t cancels the count and every noise level contributes equally.
    # Normalizing by n_masked instead would leave a residual 1/t, making lightly
    # masked (easy, high-context) examples dominate the gradient — starving the
    # fully-masked regime that every generated page starts from.
    inv_t = mx.broadcast_to(1.0 / t, (B, T)).reshape(B * T)
    return (masked_loss * inv_t).sum() / (B * T)


def train(corpus_path: str, status_callback=None):
    """Train and return (model, tokenizer). status_callback(step, total, loss, sample)
    is called periodically with the model's current output for the training display."""
    tokens, tokenizer = load_corpus(corpus_path)
    n_tokens = len(tokens)
    if n_tokens <= C.SEQ_LEN + 1:
        raise ValueError("File is too short to learn from — try a longer text.")

    # Scale steps to corpus size: more data → more steps, capped.
    base = int(n_tokens / 20)
    n_steps = max(C.MIN_STEPS, min(C.MAX_STEPS, base))

    model = DiffusionLM(tokenizer.vocab_size)
    mx.eval(model.parameters())
    print(f"Parameters: {n_params(model):,}  |  Steps: {n_steps}  |  "
          f"Corpus: {n_tokens:,} tokens  |  Vocab: {tokenizer.vocab_size}")

    # Warmup then a sustained rate: the denoiser sits at a unigram plateau for a
    # while before it starts using context, and decaying the rate too early
    # freezes it there.
    warm = optim.linear_schedule(1e-5, C.LEARNING_RATE, C.WARMUP_STEPS)
    cosine = optim.cosine_decay(C.LEARNING_RATE, max(1, n_steps - C.WARMUP_STEPS), 1e-5)
    lr_schedule = optim.join_schedules([warm, cosine], [C.WARMUP_STEPS])
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=C.WEIGHT_DECAY)

    def _loss(model, x):
        return loss_fn(model, x, tokenizer.mask_id)

    loss_and_grad = nn.value_and_grad(model, _loss)
    batcher = make_batches(tokens, C.BATCH_SIZE, C.SEQ_LEN)

    ema_loss = None
    for step in range(1, n_steps + 1):
        x = next(batcher)
        loss, grads = loss_and_grad(model, x)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        raw = float(loss)
        ema_loss = raw if ema_loss is None else 0.9 * ema_loss + 0.1 * raw

        if step % 100 == 0 or step == 1:
            print(f"  step {step}/{n_steps}  loss(ema)={ema_loss:.4f}")
            if status_callback:
                sample = quick_sample(model, tokenizer)
                status_callback(step, n_steps, ema_loss, sample)

    return model, tokenizer


def quick_sample(model, tokenizer, steps: int = 12) -> str:
    """Fast low-quality sample used only for the training display."""
    model.eval()
    T = C.SEQ_LEN
    seq = np.full(T, tokenizer.mask_id, dtype=np.int32)
    per_step = max(1, T // steps)

    for _ in range(steps):
        logits = model(mx.array(seq.reshape(1, T)))[0]
        logits_np = np.array(logits)
        # MASK/PAD are never valid outputs.
        logits_np[:, tokenizer.n_tokens:] = -np.inf
        probs = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)

        is_mask = seq == tokenizer.mask_id
        reveal_count = min(per_step, int(is_mask.sum()))
        if reveal_count == 0:
            break

        gumbel = -np.log(-np.log(np.random.uniform(1e-10, 1.0, probs.shape)))
        sampled = np.argmax(np.log(probs + 1e-30) + gumbel, axis=-1).astype(np.int32)
        conf = np.where(is_mask, probs.max(axis=-1), -np.inf)
        top_idxs = np.argsort(conf)[-reveal_count:]
        seq[top_idxs] = sampled[top_idxs]

    model.train()
    return "".join("·" * C.DISPLAY_TOKEN_W if t == tokenizer.mask_id
                   else tokenizer.decode_token(int(t)) for t in seq)
