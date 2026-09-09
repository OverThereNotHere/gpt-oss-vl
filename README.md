# gpt-oss-vl

A vision-language model built by **bolting a translator onto a frozen
20B reasoning model** — no new architecture, no new tokens (well, technically), no touching the
brain's weights.

## What this is

```
image → SigLIP2-so400m (frozen eye, 428M)
      → 2-layer MLP adapter (16.5M, TRAINED)      ← "the translator"
      → 256 vectors in gpt-oss's own embedding space
      → gpt-oss-20b (frozen brain, 20.9B MoE)     ← does the thinking
      + LoRA inserts on attention q/k/v/o (7.96M, TRAINED)  ← stage 2
```

The splice is LLaVA-style: render the harmony chat prompt with an
`{{IMAGE}}` marker, split it, embed both halves, and concatenate the image
vectors in between. No special tokens, no modified attention, nothing inside
gpt-oss was ever re-designed — the transformer is a universal *continuer of
vector conversations*, so anything placed in its embedding space gets
processed like text. (Proof from our own logs: the same brain fed random
junk vectors produces noise; fed trained-adapter vectors, it reads
diagrams.)

## Results (honest version)

- Stage 1 (adapter only): training loss **7.52 → 1.13** in 1,250 steps.
  Smoke test: the frozen brain opened a harmony analysis channel and
  reasoned about a parallelogram using point labels that exist *only in the
  image*.
- Stage 2 (+LoRA on attention): loss 1.10 → 0.96. Text-channel discipline
  improved sharply; image descriptions got concretely grounded (it read the
  name "Sultan" off a classroom whiteboard).
- 24-question trap exam: **~40% VQA accuracy** across both stages — above
  chance, below shipped LLaVA-class models. Honest verdict: a real working
  VLM prototype that "sees the gist, misses the details."
- **Live A/B on a 16GB consumer GPU:** same map, same question, greedy, only
  the LoRA toggled — with inserts: correct "United States of America" with
  grounded map-reading; without: hallucinated geography collapsing into a
  repetition loop. The trained inserts are load-bearing for grounding and
  channel stability (but don't reliably aim at truth — an orange cat comes
  out "pig"/"rabbit"/"white" depending on toggles and temperature; an
  orange sunset sky reads "blue": the diagram-diet adapter never learned
  object color). Full profile in the HF model card.

## Interesting tidbits

- **24.5M trained parameters = 0.12% of the stack.** The other 99.88% is
  frozen OpenAI + Google weights doing their day jobs.
- **Total compute bill: ~$20 on Modal** (A100-80GB, ~7 GPU-hours across
  everything, including every failed launch) — plus **$0.40 on OpenRouter**
  for the coding agent (Pi using GLM5.3-Flash) that wrote the whole thing.
- gpt-oss's native MXFP4 quantization means the brain checkpoint is *smaller
  than a Q5 requant of itself would be*. We kept it untouched.
- The eye outputs **256 native-aspect patch tokens at 448px** (SigLIP2
  NaFlex, patchified), not a fixed grid — variable token counts per image.

## The two halves

Code and weights are split across two hosts, and **each is useless without
the other**:

- **GitHub (this repo)** — the runner and the story. Without the weights
  it's a car manual with no car.
- **Hugging Face (<hf-repo-url>)** — the three weights pieces. Without the
  code they're three inert piles of numbers.

## Weights

| File | What it is |
|---|---|
| `gpt-oss-20b/` | unmodified OpenAI checkpoint (brain) |
| `siglip2-so400m-naflex/` | unmodified Google checkpoint (eye) |
| `stage2_final.pt` | our trained artifacts: `{"adapter": 16.5M, "lora": 96 A/B inserts, "step": 1250}` |

Weights live on Hugging Face (GitHub's 100MB limit makes them a non-starter
here).


## Credits

Built in one weekend for ~$20.50, mostly by asking "but *why* does that
work?" until the answers got interesting. Code by GLM5.3-Flash via Pi
