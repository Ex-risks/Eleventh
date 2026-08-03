# Eleventh

Eleventh is an experiment in masked diffusion: a generative method for discrete data that applies the logic of diffusion models to sequences of tokens rather than to pixels. Drag a plain text file onto the white window. The model learns that file, training weights from scratch — and after twenty minutes begins producing pages in its voice, ad infinitum. 

The name comes from the training corpus. *De architectura*, Vitruvius' treatise on building and the only architectural work to survive intact from antiquity, was written in ten books. Trained on all ten, the model extends it, reaching for an eleventh book that reads like the author's hand continuing past where he stopped. Ten books read closely and understood not at all.

## Reverse corruption

Autoregressive models write the way we read: one token after the next, left to right. Masked diffusion arrives differently. A page begins as noise, and the model revises every position at once until the noise resolves into text. It arrives all at once, badly, and then improves. In other words, training corrupts; generation undoes the corruption.

The model is a bidirectional transformer of about twelve million parameters — six layers, trained on a single file, on a MacBook Pro, in MLX. Roughly one ten-thousandth the size of the models people mean when they say model. That is the point. Small enough to fail legibly.

A hundred thousand words is negligible by training standards, and the output shows it: locally convincing, globally unmoored. It reproduces the source's vocabulary and cadence exactly while making claims that decay across a paragraph. I treat that as the result. Nothing it produces can be traced to text it saw elsewhere, because there is no elsewhere.

Training hides tokens behind a `[mask]` symbol — sometimes a tenth of a passage, sometimes nearly all of it — and asks the model to restore them. It sees the whole passage at once and reasons inward from both directions. A few thousand repetitions and it learns the shape of the hole a missing word leaves.

Generation runs the process backwards. The page starts fully masked: four hundred and fifty characters of nothing. At each step the model proposes a token for every empty position and reports a confidence. I keep the confident ones, freeze them, re-mask the rest, and run again with the frozen tokens as context.

Reveals follow a cosine schedule — sparse early, when the page is empty and every guess approaches a coin flip; dense late, once the surrounding text argues for what belongs in the gaps. Ninety-six steps and the page is full.

The loop is the mechanism. Predicting all positions simultaneously treats them as independent, which they are not: *the* and *circumference* are each plausible in many slots and jointly plausible in few. Committing only the confident tokens, then re-running, lets each round condition on the last.

What appears on screen is the inference itself, at the speed it happens.

## Productive failures

The first version worked on raw bytes — 256 characters, the alphabet as the alphabet. It produced fluent gibberish for weeks: letters in the right proportions, spaces in plausible places, no words.

I assumed it was undertrained and gave it more steps. Nothing changed. The loss curve was flat, and I read the flatness as a wall. It was not a wall. Measured properly, the model had learned letter frequencies and stopped, and the number I was watching could not have told me otherwise: it averaged over every corruption level, including those where the page is entirely hidden and nothing better than letter frequency is available. Only masked positions produce loss, so their count varies with *t*. Weighting each by 1/*t* cancels the variation and gives every corruption rate equal influence on the gradient.

So I measured what the model could do with context. Given ninety percent of a passage, it predicted the missing tenth at 3.00 nats per byte. A bigram lookup table — a count of which letter follows which, no learning at all — does it in 2.34. The model was losing to a lookup table, which is a clarifying kind of failure.

Then I gave it a task that required attention and nothing else. It solved that to four decimal places. The model was fine. It was spelling. Every scrap of capacity went into learning that `t-h-e` is a word, with nothing left for what words do to each other. At character level, a book this size is too little evidence for too hard a job.

The fix was to stop making it spell. The machine now learns a two-thousand-fragment vocabulary from the dropped-in book before training begins; it takes about a second. Sequences shorten by a factor of three and a half, every prediction gets easier, and the model crosses in twenty minutes a distance it could not cross in hours.

The training log records the moment. For three thousand steps the loss sits just under the score of a model that has learned only how often each word appears. Then it falls away, and the samples become sentences.

## Where it stands

It works, and it is early. Fed the *Ten Books*, it returns pages about compasses and circumferences, the planes of water, the parts of a ship — recognisably the book, recognisably not the book. It repeats itself. The grammar frays at the edges. The loss was still falling when I stopped it, which means twenty minutes is a budget, not a limit.

Open questions: how small the vocabulary can get before the voice goes with it, and whether a page can be seeded from the tail of the one before it, so the machine generates something continuous rather than a stack of unrelated pages (although that seems fun as well). 

## Running it

Double-click `launch.command`. A white window opens. Drag any `.txt` file onto it. `F` toggles fullscreen. Apple silicon only — it is built on MLX.

---

### References

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
