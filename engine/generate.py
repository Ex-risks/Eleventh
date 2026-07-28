"""Reverse (denoising) process for ELEVENTH.

Starts from a page of all [MASK] tokens and iteratively reveals positions,
highest-confidence first (MaskGIT schedule). Yields one frame dict after every
denoising step — the frame is a list of {char, masked: bool} objects that the
frontend renders in place with no reflow.

The model works in BPE tokens, but the frontend needs a CONSTANT number of
single-character cells. So each masked token renders as DISPLAY_TOKEN_W dots and
the whole canvas is padded/truncated to a fixed character width. Because the
placeholder is about as wide as an average token, the page starts near its final
size and text settles into place rather than visibly growing.
"""

import numpy as np
import mlx.core as mx

import config as C


PLACEHOLDER = "·"  # Shown at masked positions on-screen
DISPLAY_CHARS = C.SEQ_LEN * C.DISPLAY_TOKEN_W


def _seq_to_frame(seq_np: np.ndarray, tokenizer) -> list:
    """Convert a (T,) token array to a fixed-length list of {char, masked} dicts."""
    cells = []
    for tid in seq_np:
        tid = int(tid)
        if tid >= tokenizer.n_tokens:      # MASK / PAD
            cells.extend({"char": PLACEHOLDER, "masked": True}
                         for _ in range(C.DISPLAY_TOKEN_W))
        else:
            cells.extend({"char": ch, "masked": False}
                         for ch in tokenizer.decode_token(tid))
        if len(cells) >= DISPLAY_CHARS:
            return cells[:DISPLAY_CHARS]
    cells.extend({"char": " ", "masked": False}
                 for _ in range(DISPLAY_CHARS - len(cells)))
    return cells


def generate_page(model, tokenizer, seed_tail=None, temperature: float = C.GEN_TEMPERATURE):
    """Yield frames (list of cell dicts) for one full page of denoising.

    seed_tail: optional np.ndarray of up to SEQ_LEN//4 tokens from the end of
    the previous page, pinned in the first positions for continuity. Currently
    unused (fresh pages per spec default); kept as a hook.
    """
    model.eval()
    T = C.SEQ_LEN
    seq_np = np.full(T, tokenizer.mask_id, dtype=np.int32)

    # Yield the initial all-masked frame.
    yield _seq_to_frame(seq_np, tokenizer)

    steps = C.GEN_STEPS

    for step in range(steps):
        logits = model(mx.array(seq_np.reshape(1, T)))[0]      # (T, V)

        # Temperature scaling before softmax.
        logits_np = np.array(logits) / max(temperature, 1e-6)

        # MASK/PAD are never valid outputs; ban them so they can neither be
        # sampled nor dominate the confidence ranking.
        logits_np[:, tokenizer.n_tokens:] = -np.inf

        # Top-k: sample only from the k most likely tokens at each position.
        k = min(C.GEN_TOP_K, tokenizer.n_tokens)
        kth = np.partition(logits_np, -k, axis=-1)[:, -k][:, None]
        logits_np = np.where(logits_np < kth, -np.inf, logits_np)

        # Vectorized categorical sampling via Gumbel-max trick.
        probs_all = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
        probs_all /= probs_all.sum(axis=-1, keepdims=True)
        gumbel = -np.log(-np.log(np.random.uniform(1e-10, 1.0, probs_all.shape)))
        sampled = np.argmax(np.log(probs_all + 1e-30) + gumbel, axis=-1).astype(np.int32)

        max_probs = probs_all.max(axis=-1)

        is_mask = seq_np == tokenizer.mask_id
        n_masked = int(is_mask.sum())
        if n_masked == 0:
            break

        # How many to reveal this step: MaskGIT cosine schedule — few reveals
        # early (little context, samples are near-unigram draws), many late.
        frac = (step + 1) / steps
        total_to_reveal = int(round(T * (1.0 - np.cos(np.pi / 2 * frac))))
        already_revealed = T - n_masked
        reveal_count = min(max(1, total_to_reveal - already_revealed), n_masked)

        # Rank masked positions by confidence; reveal top-k.
        conf = np.where(is_mask, max_probs, -np.inf)
        top_idxs = np.argsort(conf)[-reveal_count:]

        seq_np[top_idxs] = sampled[top_idxs]

        yield _seq_to_frame(seq_np, tokenizer)

    # Final fully-revealed frame (belt-and-suspenders).
    yield _seq_to_frame(seq_np, tokenizer)
    model.train()
