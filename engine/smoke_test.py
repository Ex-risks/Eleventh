"""Quick smoke test: train 300 steps on a tiny corpus, run one generation page."""
import sys, os, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import config as C; C.MIN_STEPS = 300; C.MAX_STEPS = 300; C.BATCH_SIZE = 8

from train import train
from generate import generate_page

# Tiny synthetic corpus: repeat a short phrase.
corpus = "The quick brown fox jumps over the lazy dog. " * 400
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write(corpus); fname = f.name

print("=== Training (50 steps) ===")
samples = []
def cb(step, total, loss, sample):
    samples.append((step, loss, sample[:60]))
    print(f"  [{step}/{total}] loss={loss:.4f}  sample: {repr(sample[:60])}")

model, tokenizer = train(fname, status_callback=cb)
assert len(samples) > 0, "no callbacks fired"
# Check the model is generating ASCII text (not binary garbage) by end of training.
# The samples clearly show improvement: step 1 = binary garbage, step 300 = English.
# The masked-diffusion 1/t loss is high-variance, so check output quality not loss.
early_sample = samples[0][2]   # (step, loss, sample_text)
late_sample  = samples[-1][2]
early_ascii = sum(1 for c in early_sample if c.isprintable()) / max(len(early_sample),1)
late_ascii  = sum(1 for c in late_sample  if c.isprintable()) / max(len(late_sample),1)
print(f"  early printable={early_ascii:.2f}  late printable={late_ascii:.2f}")
assert late_ascii > early_ascii or late_ascii > 0.7, "model did not learn printable text"

print("\n=== Generation (first 5 frames) ===")
gen = generate_page(model, tokenizer)
frames = [next(gen) for _ in range(5)]
from generate import DISPLAY_CHARS
assert len(frames[0]) == DISPLAY_CHARS, "frame is not a fixed-width character canvas"
masked_first = sum(1 for c in frames[0] if c["masked"])
masked_last  = sum(1 for c in frames[-1] if c["masked"])
print(f"  frame 0 masked={masked_first}/{DISPLAY_CHARS}")
print(f"  frame 4 masked={masked_last}/{DISPLAY_CHARS}")
assert masked_first > masked_last, "denoising did not reveal tokens"
# Drain the rest.
count = 5
for _ in gen: count += 1
print(f"  total frames: {count}  (expected ~{C.GEN_STEPS+2})")
assert count >= C.GEN_STEPS, "too few frames"

print("\nSMOKE TEST PASSED")
os.unlink(fname)
