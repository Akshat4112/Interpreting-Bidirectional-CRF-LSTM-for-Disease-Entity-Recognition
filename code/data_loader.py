"""
Unified IOB/CoNLL-2003 reader for NCBI Disease and BC5CDR.

The NCBI Disease files in this repo use a non-standard tag format
(`|B-DISEASE\\n`, `|I-DISEASE\\n`, `|O\\n`); this module normalises every
tag to the canonical `B-DISEASE` / `I-DISEASE` / `O` form so that
seqeval and HuggingFace tokenisers can consume it.

A "sentence" is a maximal run of non-blank lines. NCBI Disease in this
repo does not insert blank lines between sentences (it uses a `.` token
as the boundary), so we fall back to splitting on `.` when no blank
lines are present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


CANONICAL_TAGS = ("O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL")


def _clean_tag(raw: str) -> str:
    t = raw.strip().lstrip("|").strip()
    if t in {"", "O"}:
        return "O"
    if "-" not in t:
        return "O"
    prefix, label = t.split("-", 1)
    return f"{prefix}-{label.upper()}"


def _split_on_periods(tokens: list[str], tags: list[str]) -> list[tuple[list[str], list[str]]]:
    sents: list[tuple[list[str], list[str]]] = []
    cur_t: list[str] = []
    cur_y: list[str] = []
    for tok, tag in zip(tokens, tags):
        cur_t.append(tok)
        cur_y.append(tag)
        if tok == ".":
            sents.append((cur_t, cur_y))
            cur_t, cur_y = [], []
    if cur_t:
        sents.append((cur_t, cur_y))
    return sents


@dataclass
class NERDataset:
    sentences: list[list[str]]
    tags: list[list[str]]
    name: str

    def label_set(self) -> list[str]:
        labels = {t for s in self.tags for t in s}
        # Stable canonical order: O first, then sorted by entity then B before I.
        ordered = ["O"] + sorted(l for l in labels if l != "O")
        return ordered

    def __len__(self) -> int:
        return len(self.sentences)


def load_iob(
    path: str,
    name: str | None = None,
    *,
    segment: str = "sentence",
) -> NERDataset:
    """Read an IOB / CoNLL-format file into an NERDataset.

    `segment` controls the granularity of the returned units:
      - "sentence" (default): split on `.` tokens. Matches the original
         repo's preprocessing and gives ~5400/~970/~960 train/dev/test
         units on NCBI Disease, which fits comfortably in `max_len=114`.
      - "document": one unit per blank-line-delimited block (i.e. one
         per abstract). Useful for transformer fine-tuning where longer
         context helps.

    Accepts tab- or whitespace-separated `token<sep>tag` lines and the
    repo's `|TAG\\n` quirk (cleaned automatically).
    """
    if name is None:
        name = os.path.basename(path)
    tokens_flat: list[str] = []
    tags_flat: list[str] = []
    docs: list[tuple[list[str], list[str]]] = []
    cur_t: list[str] = []
    cur_y: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped.strip() == "":
                if cur_t:
                    docs.append((cur_t, cur_y))
                    cur_t, cur_y = [], []
                continue
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(parts) < 2:
                continue
            tok, tag = parts[0], parts[-1]
            tag = _clean_tag(tag)
            cur_t.append(tok)
            cur_y.append(tag)
            tokens_flat.append(tok)
            tags_flat.append(tag)
        if cur_t:
            docs.append((cur_t, cur_y))

    if segment == "document":
        units = docs
    elif segment == "sentence":
        units = []
        for doc_t, doc_y in docs:
            units.extend(_split_on_periods(doc_t, doc_y))
        if not units and tokens_flat:
            units = _split_on_periods(tokens_flat, tags_flat)
    else:
        raise ValueError(f"unknown segment: {segment!r}")

    sentences = [t for t, _ in units]
    tags = [y for _, y in units]
    return NERDataset(sentences=sentences, tags=tags, name=name)


def load_ncbi_disease(
    root: str = "../data/ner-disease",
    *,
    segment: str = "sentence",
) -> dict[str, NERDataset]:
    return {
        "train": load_iob(os.path.join(root, "train.iob"), name="ncbi-train", segment=segment),
        "dev": load_iob(os.path.join(root, "dev.iob"), name="ncbi-dev", segment=segment),
        "test": load_iob(os.path.join(root, "test.iob"), name="ncbi-test", segment=segment),
    }


def load_bc5cdr(
    root: str = "../data/bc5cdr",
    *,
    segment: str = "sentence",
) -> dict[str, NERDataset]:
    """Load BC5CDR if present locally.

    BC5CDR is distributed by BioCreative; we don't redistribute it. Place
    `train.tsv`, `dev.tsv`, `test.tsv` (BIO format, two columns) under
    `<root>/`. The HuggingFace dataset `tner/bc5cdr` works too — convert
    to BIO with `scripts/convert_bc5cdr.py` (see README).
    """
    candidates = {
        "train": ["train.tsv", "train.iob", "train.txt"],
        "dev": ["dev.tsv", "devel.tsv", "dev.iob", "valid.tsv"],
        "test": ["test.tsv", "test.iob"],
    }
    out: dict[str, NERDataset] = {}
    for split, names in candidates.items():
        for n in names:
            p = os.path.join(root, n)
            if os.path.exists(p):
                out[split] = load_iob(p, name=f"bc5cdr-{split}", segment=segment)
                break
    if not out:
        raise FileNotFoundError(
            f"No BC5CDR files found under {root}. See README for setup."
        )
    return out


def load_conll2003(
    root: str = "../data/conll2003",
    *,
    segment: str = "document",
) -> dict[str, NERDataset]:
    """Load the CoNLL-2003 English NER corpus.

    CoNLL-2003 is the canonical general-domain comparison corpus. It is
    not redistributed in this repo (LDC-licensed); place the files at
    `<root>/{train,valid,test}.txt` in CoNLL-2003 format (token, POS,
    chunk, NER tag - whitespace-separated, blank lines between
    sentences). The HuggingFace dataset `conll2003` works too; convert
    with `scripts/convert_conll2003.py` (see README).

    Default `segment="document"` keeps the native CoNLL sentence
    boundaries (each blank-separated block is one sentence). Use
    `segment="sentence"` to additionally split on `.` (rarely useful).

    Tags returned are in IOB2 (HuggingFace conll2003 already provides
    IOB2; if your source is IOB1 with `I-X` opening a span, run it
    through `iob1_to_iob2()` below).
    """
    candidates = {
        "train": ["train.txt", "train.tsv", "train.iob"],
        "dev": ["valid.txt", "dev.txt", "valid.tsv", "dev.tsv"],
        "test": ["test.txt", "test.tsv", "test.iob"],
    }
    out: dict[str, NERDataset] = {}
    for split, names in candidates.items():
        for n in names:
            p = os.path.join(root, n)
            if os.path.exists(p):
                ds = load_iob(p, name=f"conll2003-{split}", segment=segment)
                ds.tags = iob1_to_iob2(ds.tags)
                out[split] = ds
                break
    if not out:
        raise FileNotFoundError(
            f"No CoNLL-2003 files found under {root}. See README for setup."
        )
    return out


def iob1_to_iob2(tag_seqs: list[list[str]]) -> list[list[str]]:
    """Convert IOB1-style tags to IOB2 (i.e., the first token of every
    entity span gets B-, subsequent tokens get I-). CoNLL-2003 ships in
    IOB1 by default; seqeval `mode='strict'` requires IOB2."""
    out: list[list[str]] = []
    for seq in tag_seqs:
        new = []
        prev = "O"
        for t in seq:
            if t == "O" or "-" not in t:
                new.append("O")
                prev = "O"
                continue
            prefix, label = t.split("-", 1)
            if prefix == "I":
                if prev == "O" or prev.split("-", 1)[1] != label:
                    new.append(f"B-{label}")
                else:
                    new.append(t)
            else:
                new.append(t)
            prev = new[-1]
        out.append(new)
    return out


def flatten(seqs: Iterable[Iterable[str]]) -> list[str]:
    return [x for s in seqs for x in s]
