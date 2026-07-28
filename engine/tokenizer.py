"""A small BPE vocabulary learned from the dropped file itself.

Byte-level modelling makes the denoiser spend all its capacity learning to spell,
which needs far more compute than this piece's "trains in minutes" budget allows.
Merging common character pairs into ~2k tokens shortens sequences ~4x and makes
each prediction far easier, so the book's voice emerges in minutes instead of hours.

Merges are learned over unique word *types* (weighted by frequency) rather than the
raw stream, which is what makes training the vocabulary take a second rather than
minutes. Pair counts are maintained incrementally via a pair -> words index.
"""

import re
from collections import Counter, defaultdict

# Split into words with leading whitespace attached (GPT-2 style), so " was" can
# become a single token. Spacing round-trips exactly through encode/decode.
_WORD_RE = re.compile(r"\s*\S+|\s+")

END = "</w>"  # not used as a real symbol; words are plain char tuples


class BPETokenizer:
    def __init__(self, merges, vocab):
        self.merges = merges                       # list of (a, b) in application order
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.vocab = vocab                         # list of token strings, index == id
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.unk_id = self.token_to_id.get(" ", 0)
        self._cache = {}

    # ── sizes ────────────────────────────────────────────────────────────────
    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def mask_id(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return len(self.vocab) + 1

    @property
    def vocab_size(self):
        return len(self.vocab) + 2

    # ── encode / decode ──────────────────────────────────────────────────────
    def _encode_word(self, word):
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word)
        while len(symbols) > 1:
            # Find the adjacent pair with the lowest merge rank.
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            symbols[best : best + 2] = [symbols[best] + symbols[best + 1]]
        ids = [self.token_to_id.get(s, self.unk_id) for s in symbols]
        self._cache[word] = ids
        return ids

    def encode(self, text):
        out = []
        for word in _WORD_RE.findall(text):
            out.extend(self._encode_word(word))
        return out

    def decode_token(self, tid):
        if 0 <= tid < len(self.vocab):
            return self.vocab[tid]
        return ""

    def decode(self, ids):
        return "".join(self.decode_token(int(i)) for i in ids)


def train_bpe(text: str, vocab_size: int = 2048) -> BPETokenizer:
    """Learn `vocab_size` tokens from `text` (base characters + merges)."""
    word_freq = Counter(_WORD_RE.findall(text))

    # Base vocabulary: every character present.
    chars = sorted({c for w in word_freq for c in w})
    vocab = list(chars)
    n_merges = max(0, vocab_size - len(vocab))

    # Each word is a mutable list of symbols, tracked by an integer key.
    words = [list(w) for w in word_freq]
    freqs = [word_freq[w] for w in word_freq]

    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    def index_word(i):
        syms, f = words[i], freqs[i]
        for j in range(len(syms) - 1):
            p = (syms[j], syms[j + 1])
            pair_counts[p] += f
            pair_to_words[p].add(i)

    def deindex_word(i):
        syms, f = words[i], freqs[i]
        for j in range(len(syms) - 1):
            p = (syms[j], syms[j + 1])
            pair_counts[p] -= f
            if pair_counts[p] <= 0:
                del pair_counts[p]

    for i in range(len(words)):
        index_word(i)

    merges = []
    for _ in range(n_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        affected = list(pair_to_words.get(best, ()))
        if not affected:
            del pair_counts[best]
            continue

        merged = best[0] + best[1]
        for i in affected:
            syms = words[i]
            if len(syms) < 2:
                continue
            deindex_word(i)
            j = 0
            out = []
            while j < len(syms):
                if j < len(syms) - 1 and (syms[j], syms[j + 1]) == best:
                    out.append(merged)
                    j += 2
                else:
                    out.append(syms[j])
                    j += 1
            words[i] = out
            index_word(i)

        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)
        merges.append(best)
        vocab.append(merged)

    return BPETokenizer(merges, vocab)
