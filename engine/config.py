"""Central knobs for ELEVENTH. The user never touches these."""

# --- Tokenizer (BPE learned from the dropped file itself) ---
# Byte-level modelling made the denoiser spend all its capacity learning to spell,
# which needs far more compute than this piece's "minutes" budget allows. A small
# BPE vocabulary shortens sequences ~3.5x and makes each prediction far easier.
# The real MASK/PAD ids and vocab size come from the trained tokenizer at runtime.
BPE_VOCAB_SIZE = 2048

# --- Model (small bidirectional transformer) ---
SEQ_LEN = 128           # in BPE tokens (~450 characters of text per page)
D_MODEL = 384
N_LAYERS = 6
N_HEADS = 6
FFN_DIM = 1536
# Single-book corpus: mild overfitting is the goal ("the book's voice"), so no dropout.
DROPOUT = 0.0

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
# Budget is scaled to file size at runtime; these bound it. Aim: minutes, the
# "evocative middle" -- clearly the book's voice, not verbatim recitation.
import os as _os
# Measured on a 570KB book: the denoiser sits near the unigram plateau until
# ~3-4k steps, then breaks through and starts using context. Below ~4000 steps the
# output is still word-salad, so that is the floor. ~8k steps ≈ 20 min; the cap
# bounds a large book to roughly half an hour.
MIN_STEPS = int(_os.environ.get("ELEVENTH_MIN_STEPS", 4000))
MAX_STEPS = int(_os.environ.get("ELEVENTH_MAX_STEPS", 12000))
MAX_TRAIN_BYTES = 2_000_000  # truncate-sample huge files before training

# --- Generation (the reverse / denoising process shown on screen) ---
GEN_STEPS = 96          # denoising steps per page
GEN_TEMPERATURE = 0.7   # sampling temperature for variety (endless, non-repeating)
GEN_TOP_K = 30          # sample only from the top-k tokens at each position
# A masked token renders as this many placeholder dots. Set near the average token
# length so the page starts at roughly its final size and text settles into place
# instead of visibly growing (the frontend needs a constant cell count).
DISPLAY_TOKEN_W = 4
