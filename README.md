# Eleventh

Eleventh is a small experiment in masked diffusion language modeling. Drag a plain text file onto the white window. The model reads that book and nothing else — no pretraining, no borrowed weights, no other language it has ever seen — and after about twenty minutes it begins to generate pages in the book's voice, and does not stop.

Most language models are autoregressive, writing the way we read — one token after the next, left to right. Masked diffusion models get there via a different route: a page starts as pure noise, and the model revises all positions at once until the noise resolves into text. It doesn't arrive left to right. It arrives all at once, badly, and then improves. 

The model itself is a bidirectional transformer of about twelve million parameters — six layers, trained from nothing on a single file, on a laptop, in MLX. It's roughly one ten-thousandth the size of the models people mean when they say "model." That's the point: small enough to fail clearly.

The title comes from the corpus itself. _De architectura_ — Vitruvius' treatise on building, the only architectural work to survive intact from antiquity — was written in ten books. Trained on all ten, the model doesn't summarise or imitate; it extends the numbering, reaching for an eleventh book that reads like the author's own hand continuing past where he stopped. 

Ten books read closely and understood not at all in the Eleventh.

## How it works

A 100k-word corpus is negligible by training standards, and the output reflects it: locally convincing, globally unmoored. It reproduces the source's vocabulary and cadence precisely while making claims that decay over a paragraph. I treat this as the result. Nothing the model produces can be traced to text it saw elsewhere. (There is no elsewhere).

Training corrupts the book on purpose. Take a passage, hide some fraction of its tokens behind a `[mask]` symbol — sometimes a tenth, sometimes nearly all of them — and ask the model to restore what was hidden. It sees the whole passage at once and has to reason inward from both directions. Do that a few thousand times and it learns the shape of the hole a missing word leaves.

Generation runs that process in reverse. The page starts entirely masked — four hundred and fifty characters of nothing. At each step the model proposes a token for every empty position and reports a confidence for each. I keep only the ones it's most sure of, freeze them, re-mask the rest, and run again with those tokens now visible as context. 

Reveals follow a cosine schedule: sparse early, when the page is empty and every guess is close to a coin flip; dense late, once the surrounding text starts arguing for what belongs in the remaining gaps. Ninety-six steps later, the page is full.

This loop is the essential mechanism. Predicting all positions at once treats them as independent, which is not the case — "the" and "circumference" are each plausible in many slots, but jointly plausible in few. Committing only the confident tokens each round, then re-running, lets every round condition on the last round's commitments.

The reveal on screen is therefore, the actual inference, happening at the speed it happens.

## Productive Failures

Some interesting things I discovered. The first version worked on raw bytes — 256 characters, the alphabet as the alphabet. It produced fluent gibberish for weeks: letters in the right proportions, spaces in plausible places, and no words.

I assumed it was undertrained and gave it more steps, which did nothing. The loss curve was flat, (I read the flatness as a wall but It wasn't). Measured properly, the model had learned the frequency of letters and stopped — and the number I was watching couldn't have told me otherwise, since it averaged over every corruption level, including the ones where the page is entirely hidden and nothing better than letter frequencies is possible. Only masked positions produce loss, so their count varies with *t*; weighting each by 1/*t* cancels that out, giving every corruption rate equal influence on the gradient.

So I measured what the model could do when it actually had context. Given ninety percent of a passage, it predicted the missing tenth at 3.00 nats per byte. A bigram lookup table — a count of which letter follows which, no learning at all — does it in 2.34. The model was losing to a lookup table, which is a clarifying kind of failure.

Then I gave it a task that required attention and nothing else, and it solved that to four decimal places. The model was fine — it was spelling. Every scrap of its capacity was going into learning that `t-h-e` is a word, with nothing left over for what words do to each other. At character level, a book this size is simply too little evidence for too hard a job.

The fix was to stop making it spell. The machine now learns a small vocabulary — two thousand fragments — from the dropped-in book itself, before training begins; it takes about a second. Sequences get three and a half times shorter, every prediction gets easier, and the model crosses in twenty minutes a distance it couldn't cross in hours.

There's a moment in the training log where this happens. For three thousand steps the loss sits just under the score of a model that has learned only how often each word appears, and then it falls away, and the samples become sentences.

## Where it stands

It works, in an interesting way, and it's early. Fed Vitruvius' *Ten Books on Architecture*, it returns pages about compasses and circumferences, about the planes of water and the parts of a ship — recognisably the book, recognisably not the book. It repeats itself. The grammar frays at the edges. The loss was still falling when I stopped it, which means the twenty minutes is a budget, not a limit.

Open questions I'm working on: how small the vocabulary can get before the voice goes with it; and whether a page can be seeded from the tail of the one before it, so the machine generates something continuous rather than a stack of unrelated pages.

## Running it

Double-click `launch.command`. A white window opens. Drag any `.txt` file onto it. `F` toggles fullscreen. Apple silicon only — it's built on MLX.

---

### References

Some references that helped me get to this point:

1. Austin, J., Johnson, D., Ho, J., Tarlow, D., van den Berg, R. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS.
2. Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W. T. (2022). *MaskGIT: Masked Generative Image Transformer.* CVPR.
3. Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., Hashimoto, T. (2022). *Diffusion-LM Improves Controllable Text Generation.* NeurIPS.
4. Lou, A., Meng, C., Ermon, S. (2024). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML.
5. Sahoo, S. et al. (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS.
6. Nie, S. et al. (2025). *Large Language Diffusion Models* (LLaDA).
7. Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
8. Shannon, C. E. (1951). *Prediction and Entropy of Printed English.* Bell System Technical Journal.
9. Hannun, A., Digani, J., Katharopoulos, A., Collobert, R. (2023). *MLX: Efficient and flexible machine learning on Apple silicon.*
10. Vitruvius. *The Ten Books on Architecture*, tr. M. H. Morgan (1914). Project Gutenberg #20239.
