"""gpt-oss-vl runner v0.2 — one-shot question answering. Self-contained.

Edit the CONFIG block at the top, then:  python run_vlm.py

Assembles the three weights pieces (brain / eye / trained-checkpoint) into
a working model and answers one question about one image. Follow-up chat,
fancier interfaces come later — one thing at a time.

Requires: torch, transformers, pillow, safetensors, requests (see
requirements.txt). Models folder layout:

    MODELS_DIR/
    ├── gpt-oss-20b/                  (brain, unmodified HF checkpoint)
    ├── siglip2-so400m-naflex/        (eye, unmodified HF checkpoint)
    └── stage2_final.pt               (the trained adapter + LoRA inserts)
"""

import math
import os

# set BEFORE torch initializes CUDA — reduces fragmentation on tight cards
# (new alias preferred by torch >= 2.8; old var kept for 2.7 compatibility)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================== CONFIG ==============================
# Folder containing the three weights pieces (the HF export).
MODELS_DIR = os.path.expanduser("~/gpt-oss-vl/models")

# Optional: path (or http(s) URL) to an image. Empty string = text-only.
IMAGE_PATH = ""

# The prompt. For images, questions that reference the figure work best
# (that's what it trained on).
QUESTION = "Testing testing, is this thing on?"

# Generation length. The model thinks in a harmony "analysis" channel
# before answering; short budgets cut it off mid-thought.
MAX_NEW_TOKENS = 512

# Sampling. 0.0 = greedy (deterministic — best for A/B tests: same inputs
# give byte-identical outputs). The checkpoint's native operating point is
# temperature 1.0, top_p 1.0 with top_k disabled — gpt-oss's default.
TEMPERATURE = 0.0
TOP_P = 1.0

# Apply the stage-2 LoRA inserts? False = stage-1 behavior (translator
# only). Useful for A/B: the trap exam showed the LoRA healed the text
# channel but sometimes made text-mode answers confabulate about images.
APPLY_LORA = True
# =====================================================================

# ---------------------------------------------------------------
# Model-specific constants (hard-coded on purpose: this runner runs
# exactly one model, so every constant is an optimization handle).
IMAGE_MARKER = "{{IMAGE}}"       # splice point in the harmony chat prompt
IMAGE_RES = 448                  # SigLIP2 NaFlex preprocessing resolution
VISION_HIDDEN = 1152             # eye output dim
ADAPTER_INTERMEDIATE = 4096      # translator MLP width
LLM_HIDDEN = 2880                # brain embedding dim
LORA_RANK, LORA_ALPHA = 16, 32   # stage-2 insert geometry
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


# ---------------------------------------------------------------
# LoRA: frozen Linear + low-rank delta, loaded from stage2_final.pt.
# (Inlined from the training code so this folder stands alone.)
class LoRALinear(nn.Module):
    """Frozen base Linear + low-rank delta (B@A, loaded nonzero here)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.scale = LORA_ALPHA / LORA_RANK
        # live on the SAME device as the base layer (crash: A/B stayed on
        # CPU while wrapped attention ran on CUDA)
        self.lora_A = nn.Parameter(torch.zeros(
            LORA_RANK, base.in_features, dtype=torch.float32,
            device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, LORA_RANK, dtype=torch.float32,
            device=base.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x.to(torch.float32) @ self.lora_A.T @ self.lora_B.T) * self.scale
        return out + delta.to(out.dtype)


def apply_lora(model: nn.Module) -> int:
    """Wrap every attention projection in LORA_TARGETS. Returns count."""
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in LORA_TARGETS:
                setattr(module, child_name, LoRALinear(child))
                replaced += 1
    return replaced


def load_lora_state_dict(model: nn.Module, sd: dict) -> None:
    """Copy saved {"A":…, "B":…} pairs into the live inserts."""
    modules = dict(model.named_modules())
    for name, ab in sd.items():
        module = modules.get(name)
        if not isinstance(module, LoRALinear):
            raise KeyError(f"lora checkpoint refers to unknown module {name!r}")
        device = module.lora_A.device
        module.lora_A.data = ab["A"].to(device=device, dtype=torch.float32)
        module.lora_B.data = ab["B"].to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------
# The translator: eye-dim -> brain-dim, loaded from stage2_final.pt.
class Adapter(nn.Module):
    """LLaVA-style 2-layer MLP: 1152 -> 4096 -> 2880."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(VISION_HIDDEN)
        self.fc1 = nn.Linear(VISION_HIDDEN, ADAPTER_INTERMEDIATE)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ADAPTER_INTERMEDIATE, LLM_HIDDEN)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # dtype boundary: tower emits bf16, adapter may hold fp32 — cast first
        x = patch_tokens.to(self.fc1.weight.dtype)
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------
def log(*args, **_kwargs):
    print(*args, flush=True)


def load_image(source):
    from PIL import Image

    if source.startswith(("http://", "https://")):
        import io

        import requests

        return Image.open(io.BytesIO(requests.get(source, timeout=30).content))
    return Image.open(source)


def main():
    from transformers import (AutoModelForCausalLM, AutoProcessor,
                              AutoTokenizer, Siglip2Model)

    brain_dir = os.path.join(MODELS_DIR, "gpt-oss-20b")
    eye_dir = os.path.join(MODELS_DIR, "siglip2-so400m-naflex")
    ckpt_path = os.path.join(MODELS_DIR, "stage2_final.pt")
    for p in (brain_dir, eye_dir, ckpt_path):
        assert os.path.exists(p), f"missing: {p}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    if device == "cpu":
        log("[runner] WARNING: no CUDA — this will be extremely slow.")

    # ---- brain ---------------------------------------------------------
    log("[runner] loading gpt-oss-20b …", flush=True)
    brain = AutoModelForCausalLM.from_pretrained(
        brain_dir, attn_implementation="eager"
    )
    brain.to(device=device, dtype=dtype)
    brain.eval()
    tok = AutoTokenizer.from_pretrained(brain_dir)
    embed_layer = brain.get_input_embeddings()

    # ---- eye + translator ---------------------------------------------
    # Keep them off the GPU if the brain left it crowded: they run once per
    # image, so CPU costs ~1s and zero quality.
    small_dev = "cuda"
    if device == "cuda":
        free_b, _total = torch.cuda.mem_get_info()
        if free_b < 3 * 1024**3:
            small_dev = "cpu"
            log(f"[runner] only {free_b/1024**3:.1f}GB VRAM free after the "
                "brain — eye+adapter go to CPU (adds ~1s per image)", flush=True)
    eye_dtype = dtype if small_dev == "cuda" else torch.float32
    log("[runner] loading SigLIP2 eye …", flush=True)
    eye_full = Siglip2Model.from_pretrained(eye_dir)
    eye = eye_full.vision_model          # vision tower ONLY — the full dual
    # encoder would demand input_ids for its text half (that was the crash);
    # same extraction the training code used
    eye = eye.to(device=small_dev, dtype=eye_dtype)
    eye.eval()
    # size=448x448 must match training preprocessing exactly
    processor = AutoProcessor.from_pretrained(
        eye_dir, size={"height": IMAGE_RES, "width": IMAGE_RES}
    )
    adapter = Adapter()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ck["adapter"])
    adapter = adapter.to(device=small_dev, dtype=eye_dtype)
    adapter.eval()

    # ---- stage-2 LoRA inserts on the brain's attention -----------------
    if APPLY_LORA:
        log("[runner] applying stage-2 LoRA …", flush=True)
        n = apply_lora(brain)
        load_lora_state_dict(brain, ck["lora"])
        log(f"[runner] {n} attention projections wrapped with trained inserts",
            flush=True)
    else:
        log("[runner] APPLY_LORA=False — running stage-1 (translator only)",
            flush=True)

    # ---- build the prompt ----------------------------------------------
    has_image = bool(IMAGE_PATH)
    if has_image:
        content = f"Look at this image.\n{IMAGE_MARKER}\n{QUESTION}"
    else:
        content = QUESTION
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    @torch.no_grad()
    def embed_prompt():
        n_img = 0
        if not has_image:
            ids = tok(rendered, return_tensors="pt",
                      add_special_tokens=False).input_ids.to(device)
            return embed_layer(ids).squeeze(0), n_img  # (T, H)
        # split at the marker, embed both halves, splice the image between
        before, after = rendered.split(IMAGE_MARKER, 1)
        img = load_image(IMAGE_PATH).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(small_dev) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=eye.dtype)
        patch = eye(**inputs).last_hidden_state.squeeze(0)   # (N, 1152)
        img_embeds = adapter(patch).to(device=device, dtype=dtype)  # (N, 2880)
        n_img = int(img_embeds.shape[0])
        pre = embed_layer(
            tok(before, add_special_tokens=False,
                return_tensors="pt").input_ids.to(device)
        ).squeeze(0)
        suf = embed_layer(
            tok(after, add_special_tokens=False,
                return_tensors="pt").input_ids.to(device)
        ).squeeze(0)
        return torch.cat([pre, img_embeds, suf], dim=0), n_img  # (T, H)

    prompt_embeds, n_img = embed_prompt()
    log(f"[runner] prompt: {prompt_embeds.shape[0]} embeds "
        f"({n_img} from image)", flush=True)

    # ---- generate --------------------------------------------------------
    log("[runner] generating …", flush=True)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS)
    if TEMPERATURE > 0:
        # native sampling mode: pure nucleus, top_k off (checkpoint default)
        gen_kwargs.update(do_sample=True, temperature=TEMPERATURE,
                          top_p=TOP_P, top_k=0)
    else:
        gen_kwargs.update(do_sample=False)
    out = brain.generate(
        inputs_embeds=prompt_embeds.unsqueeze(0),
        attention_mask=torch.ones(1, prompt_embeds.shape[0], dtype=torch.long,
                                  device=device),
        **gen_kwargs,
    )
    text = tok.decode(out[0], skip_special_tokens=False)

    print("\n" + "=" * 60)
    print("RAW MODEL OUTPUT (harmony format)")
    print("=" * 60)
    print(text)
    print("=" * 60)
    if "<|channel|>final<|message|>" in text:
        answer = text.split("<|channel|>final<|message|>")[1]
        answer = answer.split("<|return|>")[0].split("<|end|>")[0].strip()
        print("\nFINAL ANSWER:\n" + answer)
    else:
        print("\n(no final channel — raise MAX_NEW_TOKENS if it was cut off)")


if __name__ == "__main__":
    main()
