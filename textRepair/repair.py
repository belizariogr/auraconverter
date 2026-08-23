#!/usr/bin/env python3
"""Tiny local instruct model to fix PDF letter-spacing / split-word extraction.

Uses Qwen2.5-0.5B-Instruct (MLX 4-bit on Apple Silicon, Transformers elsewhere).
Reads JSON {"chunks": ["..."]} on stdin, prints JSON {"chunks": ["..."]} on stdout.
"""

from __future__ import annotations

import json
import os
import sys

SYSTEM = (
    "You repair text extracted from PDF or EPUB. "
    "Fix only letters and words split by layout (example: 'p a l a v r a' → 'palavra', "
    "'infor-\\nmação' → 'informação'). "
    "Keep the original language, punctuation, line breaks, names, and meaning. "
    "Do not summarize, explain, translate, or invent. "
    "Return only the corrected text."
)

MLX_MODEL = os.environ.get(
    "TEXT_REPAIR_MLX_MODEL",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
)
HF_MODEL = os.environ.get(
    "TEXT_REPAIR_HF_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct",
)


def _cache_dir() -> str | None:
    return os.environ.get("TEXT_REPAIR_CACHE") or None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def repair_mlx(chunks: list[str]) -> list[str]:
    from mlx_lm import generate, load

    cache = _cache_dir()
    kwargs = {"tokenizer_config": {"trust_remote_code": True}}
    if cache:
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache)

    model, tokenizer = load(MLX_MODEL)

    sampler = None
    try:
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.1)
    except Exception:
        sampler = None

    out: list[str] = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": chunk},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = SYSTEM + "\n\n" + chunk + "\n"

        max_tokens = min(2048, max(64, int(len(chunk) * 0.8) + 32))
        gen_kwargs = {"prompt": prompt, "max_tokens": max_tokens, "verbose": False}
        if sampler is not None:
            gen_kwargs["sampler"] = sampler
        else:
            gen_kwargs["temp"] = 0.1

        raw = generate(model, tokenizer, **gen_kwargs)
        if isinstance(raw, str) and prompt and raw.startswith(prompt):
            raw = raw[len(prompt) :]
        out.append(_strip_fences(str(raw)))
    return out


def repair_hf(chunks: list[str]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = _cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, cache_dir=cache, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        cache_dir=cache,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    out: list[str] = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": chunk},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        max_new = min(2048, max(64, int(len(chunk) * 0.8) + 32))
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
            )
        gen = ids[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(gen, skip_special_tokens=True)
        out.append(_strip_fences(raw))
    return out


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"chunks": []}))
        return 0
    data = json.loads(raw)
    chunks = [str(c) for c in (data.get("chunks") or [])]
    if not chunks:
        print(json.dumps({"chunks": []}))
        return 0

    try:
        import mlx_lm  # noqa: F401

        repaired = repair_mlx(chunks)
    except ImportError:
        try:
            repaired = repair_hf(chunks)
        except Exception as exc:
            print(f"text-repair backend unavailable: {exc}", file=sys.stderr)
            return 2

    print(json.dumps({"chunks": repaired}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
