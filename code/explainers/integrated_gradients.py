"""
Integrated Gradients for HuggingFace token-classification models.

Wraps Captum's `LayerIntegratedGradients` against the model's input
embeddings. For a target sub-token position, we attribute the model's
predicted-class logit back to each input embedding row, then sum the
embedding-axis attributions to get a per-sub-token scalar, then map
back to per-word scalars via the same word_ids() alignment used by
`train_biobert.py`.

Output shape matches LIME / attention attributions: a list of length
`len(words)` aligned token-for-token with the original sentence.

Requires: torch, captum, transformers. GPU strongly recommended for
batch IG; the function falls back to CPU if `device='cpu'` is passed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def integrated_gradients_word_attribution(
    model,
    tokenizer,
    words: Sequence[str],
    target_word_idx: int,
    *,
    target_label_id: int | None = None,
    n_steps: int = 25,
    max_length: int = 192,
    device: str | None = None,
):
    """Per-word IG attribution for the prediction at `target_word_idx`.

    If `target_label_id` is None, attribute against the model's argmax
    label at the target position.
    """
    import torch
    from captum.attr import LayerIntegratedGradients

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    enc = tokenizer(
        list(words),
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = enc.word_ids()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    target_token_pos = None
    for tok_pos, wid in enumerate(word_ids):
        if wid == target_word_idx:
            target_token_pos = tok_pos
            break
    if target_token_pos is None:
        return [0.0] * len(words)

    if target_label_id is None:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        target_label_id = int(logits[0, target_token_pos].argmax().item())

    def forward(ids):
        out = model(input_ids=ids, attention_mask=attention_mask).logits
        return out[:, target_token_pos, target_label_id]

    embed_layer = _get_word_embeddings(model)
    lig = LayerIntegratedGradients(forward, embed_layer)
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        n_steps=n_steps,
        return_convergence_delta=False,
    )
    # (1, seq, hidden) -> (seq,) by L2 over hidden, then map subtokens to words
    per_token = attributions.detach().cpu().numpy()[0]
    per_token = np.linalg.norm(per_token, ord=2, axis=-1)

    per_word = [0.0] * len(words)
    counts = [0] * len(words)
    for tok_pos, wid in enumerate(word_ids):
        if wid is None or wid >= len(words):
            continue
        per_word[wid] += float(per_token[tok_pos])
        counts[wid] += 1
    return [s / max(c, 1) for s, c in zip(per_word, counts)]


def _get_word_embeddings(model):
    """Return the input word-embedding layer regardless of architecture."""
    for path in ("bert.embeddings.word_embeddings",
                 "roberta.embeddings.word_embeddings",
                 "deberta.embeddings.word_embeddings",
                 "electra.embeddings.word_embeddings",
                 "embeddings.word_embeddings"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError("Could not locate word_embeddings layer on model")
