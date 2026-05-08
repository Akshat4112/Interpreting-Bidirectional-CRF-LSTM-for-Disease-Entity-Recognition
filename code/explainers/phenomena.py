"""
Bucket sentences by linguistic phenomena that distinguish biomedical
text from general English. Used to condition disagreement metrics on
the phenomena that actually drive the cross-domain gap.

Phenomena:

  - oov_rate(words, vocab): fraction of tokens not in `vocab` (a
    reference general-English vocabulary, e.g. all words in CoNLL-2003
    train + a frequency cap from a Wikipedia dump). Higher = more
    domain-specific.

  - has_multiword_entity(tags): True if any entity span spans 2+ tokens
    (B- followed by I- ...). NCBI Disease has many ("adenomatous
    polyposis coli tumour"), CoNLL-2003 has fewer per-token.

  - abbreviation_density(words, tags): fraction of entity tokens that
    look like abbreviations - all caps, hyphens, length <= 5, often
    parenthesised, e.g. "A-T", "ATM", "APC". Common in biomedical text,
    rare in general news.

  - contains_paren_definition(words): True if the sentence contains a
    "Disease Name ( ABBR )" pattern, which is dense in biomedical text.

`bucketize()` takes a list of sentences with their tags and a reference
vocabulary, and returns a dict of bucket -> sentence indices, plus
per-sentence metadata.
"""

from __future__ import annotations

import re
from typing import Sequence


_ABBR_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,4}$")


def oov_rate(words: Sequence[str], vocab: set[str]) -> float:
    if not words:
        return 0.0
    n_oov = sum(1 for w in words if w.lower() not in vocab)
    return n_oov / len(words)


def has_multiword_entity(tags: Sequence[str]) -> bool:
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            label = t.split("-", 1)[1]
            if i + 1 < len(tags) and tags[i + 1] == f"I-{label}":
                return True
    return False


def num_entities(tags: Sequence[str]) -> int:
    return sum(1 for t in tags if t.startswith("B-"))


def abbreviation_density(words: Sequence[str], tags: Sequence[str]) -> float:
    """Fraction of *entity* tokens that look like abbreviations."""
    ent_words = [w for w, t in zip(words, tags) if t != "O"]
    if not ent_words:
        return 0.0
    return sum(1 for w in ent_words if _ABBR_RE.match(w)) / len(ent_words)


def contains_paren_definition(words: Sequence[str]) -> bool:
    """Heuristic: look for `( <abbr> )` token sequence."""
    for i in range(1, len(words) - 1):
        if words[i - 1] == "(" and words[i + 1] == ")" and _ABBR_RE.match(words[i]):
            return True
    return False


def per_sentence_features(
    sentences: Sequence[Sequence[str]],
    tags: Sequence[Sequence[str]],
    *,
    vocab: set[str],
) -> list[dict]:
    rows = []
    for i, (words, ts) in enumerate(zip(sentences, tags)):
        rows.append({
            "sentence_id": i,
            "n_tokens": len(words),
            "n_entities": num_entities(ts),
            "oov_rate": oov_rate(words, vocab),
            "has_multiword_entity": has_multiword_entity(ts),
            "abbreviation_density": abbreviation_density(words, ts),
            "contains_paren_definition": contains_paren_definition(words),
        })
    return rows


def bucketize(
    features: list[dict],
    *,
    oov_threshold: float = 0.4,
    abbr_threshold: float = 0.2,
) -> dict[str, list[int]]:
    """Return {bucket_name: [sentence_id, ...]} for the buckets used in
    the disagreement table. Sentences may appear in multiple buckets."""
    out: dict[str, list[int]] = {
        "all": [], "high_oov": [], "low_oov": [],
        "has_multiword_entity": [], "single_token_entities_only": [],
        "high_abbreviation": [], "has_paren_definition": [],
    }
    for r in features:
        i = r["sentence_id"]
        out["all"].append(i)
        if r["oov_rate"] >= oov_threshold:
            out["high_oov"].append(i)
        else:
            out["low_oov"].append(i)
        if r["has_multiword_entity"]:
            out["has_multiword_entity"].append(i)
        else:
            out["single_token_entities_only"].append(i)
        if r["abbreviation_density"] >= abbr_threshold:
            out["high_abbreviation"].append(i)
        if r["contains_paren_definition"]:
            out["has_paren_definition"].append(i)
    return out


def vocab_from_dataset(sentences: Sequence[Sequence[str]]) -> set[str]:
    """Lower-cased token vocabulary from a reference corpus (e.g.,
    CoNLL-2003 train) used as the OOV reference set."""
    return {w.lower() for s in sentences for w in s}
