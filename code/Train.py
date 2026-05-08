"""
BiLSTM and BiLSTM-CRF training for disease NER.

Two changes vs. the original Train.py:

1. There is now a real BiLSTM-CRF variant. The CRF layer is implemented
   in `crf_layer.py` (no `keras-contrib` / `tensorflow-addons`
   dependency, both of which are unmaintained as of 2025).

2. Evaluation runs on the held-out NCBI test set (test.iob), not on a
   train-internal validation split, and uses entity-level seqeval
   metrics. The previous setup never touched test.iob.

Seed control is exposed via `seed=` so the multi-seed runner can
aggregate mean ± std.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from collections import Counter
from typing import Any

import numpy as np

from data_loader import NERDataset
from Evaluation import evaluate

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.utils import pad_sequences
    _TF_OK = True
except Exception as _e:  # pragma: no cover
    _TF_OK = False
    _TF_IMPORT_ERR = _e


def _require_tf() -> None:
    if not _TF_OK:
        raise ImportError(
            f"TensorFlow is not available in this environment ({_TF_IMPORT_ERR}). "
            "Install tensorflow>=2.10 to train the BiLSTM/BiLSTM-CRF models."
        )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TF_OK:
        tf.random.set_seed(seed)


class Encoder:
    """Builds and applies (word2idx, tag2idx) over an `NERDataset`."""

    def __init__(self, max_len: int = 114, vocab_size: int = 5000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.word2idx: dict[str, int] = {}
        self.tag2idx: dict[str, int] = {}

    def fit(self, train: NERDataset) -> "Encoder":
        word_cnt: Counter[str] = Counter()
        for s in train.sentences:
            word_cnt.update(s)
        most_common = [w for w, _ in word_cnt.most_common(self.vocab_size)]
        self.word2idx = {"PAD": 0, "UNK": 1}
        for i, w in enumerate(most_common, start=2):
            self.word2idx[w] = i
        labels = train.label_set()
        self.tag2idx = {t: i for i, t in enumerate(labels)}
        return self

    @property
    def idx2tag(self) -> dict[int, str]:
        return {i: t for t, i in self.tag2idx.items()}

    def transform(self, ds: NERDataset) -> tuple[np.ndarray, np.ndarray]:
        _require_tf()
        unk = self.word2idx["UNK"]
        pad_w = self.word2idx["PAD"]
        pad_y = self.tag2idx.get("O", 0)
        X = [[self.word2idx.get(w, unk) for w in s] for s in ds.sentences]
        y = [[self.tag2idx.get(t, pad_y) for t in ts] for ts in ds.tags]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=pad_w)
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=pad_y)
        return X, y

    def decode_predictions(
        self,
        ds: NERDataset,
        pred_ids: np.ndarray,
    ) -> list[list[str]]:
        """Trim padded predictions back to original sentence lengths."""
        idx2tag = self.idx2tag
        out: list[list[str]] = []
        for sent, row in zip(ds.sentences, pred_ids):
            n = min(len(sent), self.max_len)
            out.append([idx2tag.get(int(row[i]), "O") for i in range(n)])
        return out

    def save(self, root: str = "../data") -> None:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "word2idx.pkl"), "wb") as f:
            pickle.dump(self.word2idx, f)
        with open(os.path.join(root, "tag2idx.pkl"), "wb") as f:
            pickle.dump(self.tag2idx, f)


def build_bilstm(n_words: int, n_tags: int, max_len: int) -> "keras.Model":
    _require_tf()
    inp = keras.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=n_words, output_dim=50, input_length=max_len, mask_zero=True)(inp)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(x)
    out = layers.TimeDistributed(layers.Dense(n_tags, activation="softmax"))(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm_crf(n_words: int, n_tags: int, max_len: int) -> "keras.Model":
    _require_tf()
    from crf_layer import CRF  # local, lazy

    inp = keras.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=n_words, output_dim=50, input_length=max_len, mask_zero=True)(inp)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(x)
    x = layers.TimeDistributed(layers.Dense(n_tags))(x)
    crf = CRF(n_tags, name="crf")
    out = crf(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=crf.loss, metrics=[crf.accuracy])
    return model


def train_and_eval(
    train_ds: NERDataset,
    test_ds: NERDataset,
    *,
    model_kind: str = "bilstm",
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    save_dir: str = "../models",
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Fit `model_kind` on `train_ds` and report seqeval metrics on `test_ds`."""
    _require_tf()
    set_global_seed(seed)

    enc = Encoder().fit(train_ds)
    enc.save()
    X_tr, y_tr = enc.transform(train_ds)
    X_te, y_te = enc.transform(test_ds)
    n_words = max(enc.word2idx.values()) + 1
    n_tags = len(enc.tag2idx)

    if model_kind == "bilstm":
        model = build_bilstm(n_words, n_tags, enc.max_len)
        y_tr_fit = y_tr.reshape(*y_tr.shape, 1)
    elif model_kind == "bilstm_crf":
        model = build_bilstm_crf(n_words, n_tags, enc.max_len)
        y_tr_fit = y_tr  # CRF loss takes 2D ints
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")

    history = model.fit(
        X_tr, y_tr_fit,
        batch_size=batch_size, epochs=epochs,
        validation_split=0.1, verbose=2,
    )

    os.makedirs(save_dir, exist_ok=True)
    name = os.path.join(save_dir, f"{model_kind}_seed{seed}_{int(time.time())}.h5")
    try:
        model.save(name)
    except Exception:
        # CRF custom layer may not pickle cleanly with .h5 in all TF versions.
        model.save_weights(name.replace(".h5", ".weights.h5"))

    raw = model.predict(X_te, batch_size=batch_size, verbose=0)
    if model_kind == "bilstm":
        pred_ids = raw.argmax(-1)
    else:
        pred_ids = raw  # CRF returns Viterbi-decoded ints
    y_pred_seqs = enc.decode_predictions(test_ds, pred_ids)
    y_true_seqs = test_ds.tags

    metrics = evaluate(y_true_seqs, y_pred_seqs, print_report=True)
    metrics["model_kind"] = model_kind
    metrics["seed"] = seed
    metrics["weights_path"] = name
    if return_predictions:
        metrics["y_pred"] = y_pred_seqs
        metrics["y_true"] = y_true_seqs
    return metrics
