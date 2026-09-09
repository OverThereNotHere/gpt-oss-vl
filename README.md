# gpt-oss-vl runner

> **Status: v0.2 — one-shot mode works.** Edit the CONFIG block at the top
> of `run_vlm.py` (models folder, optional image path/URL, question), run
> `python run_vlm.py`, get the raw harmony output plus the final answer.
> No CLI flags, no chat loop yet — deliberate, one thing at a time.
> Chat REPL, follow-ups and URL polish are the planned v0.3.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pins matter: `transformers>=4.55` (gpt-oss architecture support) and
`triton>=3.4` (native MXFP4 kernels). With both, the brain loads in native
4-bit (~13GB VRAM, same league as llama.cpp). If triton is missing,
transformers prints a warning and silently dequantizes to bf16 (~41GB) —
that warning is the thing to watch for on first run.

## Recommended settings

In `run_vlm.py`'s CONFIG block:

| Use case | TEMPERATURE | TOP_P | Notes |
|---|---|---|---|
| Image questions | **0.0** (greedy) or ≤ 0.4 | 1.0 | deterministic; best answers; short outputs don't hit repetition risk |
| Chat (no image) | ~0.7 | ~0.95 | conventional assistant feel |
| "Feel the entropy" | 1.0 | 1.0 | the checkpoint's native operating point — wide scatter on visual questions |

Greedy is also the scientific instrument: same inputs give byte-identical
outputs, so toggling `APPLY_LORA` isolates exactly what the trained inserts
change.

## What this is

A small, dependency-light script (`run_vlm.py`) that assembles the three
weights pieces into a working model and answers questions about images:

```
eye (SigLIP2-so400m)  →  translator (adapter + LoRA)  →  brain (gpt-oss-20b)
        428M params              16.5M + 7.96M                20.9B, frozen
```

Planned usage (interface draft — subject to the actual implementation):

```bash
pip install -r requirements.txt   # torch, transformers, pillow, safetensors

python run_vlm.py \
  --models-dir ~/gpt-oss-vl/models \
  --image path/to/image.png \
  --question "What is shown in this image?"
```

## What it does under the hood (so you can port it)

1. Loads gpt-oss-20b (bf16 on GPU) and SigLIP2 (NaFlex, native aspect ratio,
   448px) via `transformers`.
2. Builds the LLaVA-style splice: renders the harmony chat prompt with an
   `{{IMAGE}}` marker, splits the string, encodes the image into **256
   patch tokens**, projects them through the adapter into gpt-oss's
   embedding space (2,880-dim), and concatenates
   `[prefix embeds | image embeds | suffix embeds]`.
3. Applies the stage-2 LoRA inserts (96 attention q/k/v/o projections).
4. Generates greedily in gpt-oss's harmony format (analysis channel +
   final channel). The raw output includes both channels; the runner
   prints the final answer.

No new tokens, no modified attention, no architectural surgery — the
transformer treats the 256 image vectors exactly like text tokens because
they live in the same vector space. If you want to reimplement this in
another engine, those four steps are the whole spec.


## Relationship to the rest of the project

- **Weights alone don't run** — they're three inert piles of numbers until
  something splices them. This runner (with the `mmv/` package it imports)
  is that something.
