"""
Minimal linear-chain CRF layer for Keras / TF 2.x.

Self-contained — no `keras-contrib` or `tensorflow-addons` (both
unmaintained as of 2025). Implements:

  - log-likelihood loss via the forward algorithm
  - Viterbi decoding at inference

Inputs to `call(unary_scores)` are emission scores of shape
`(batch, max_len, n_tags)`. The layer adds learned transition scores of
shape `(n_tags, n_tags)` plus start / end vectors. During training the
forward algorithm computes `log Z`; the loss is `-log P(y | x)`.

Padding is detected via the propagated Keras mask (set
`mask_zero=True` on the embedding upstream).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class CRF(layers.Layer):
    def __init__(self, n_tags: int, **kwargs):
        super().__init__(**kwargs)
        self.n_tags = n_tags
        self.supports_masking = True

    def build(self, input_shape):
        self.transitions = self.add_weight(
            name="transitions",
            shape=(self.n_tags, self.n_tags),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # Cache potentials + mask so `loss` and `accuracy` can recover them
        # without re-threading them through the Keras API. This pattern is
        # intentional - it avoids needing a custom Model subclass.
        self._last_potentials = inputs
        self._last_mask = mask
        seq_len = self._sequence_lengths(inputs, mask)
        return self._viterbi_decode(inputs, seq_len)

    # ------------------------------------------------------------------
    def _sequence_lengths(self, inputs, mask):
        if mask is None:
            return tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
        return tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)

    def _log_norm(self, potentials, seq_len):
        # Forward algorithm; potentials: (B, T, N).
        first = potentials[:, 0, :]
        rest = potentials[:, 1:, :]

        def step(prev, cur):
            # prev: (B, N), cur: (B, N)
            broadcast = tf.expand_dims(prev, 2) + self.transitions  # (B, N, N)
            return cur + tf.reduce_logsumexp(broadcast, axis=1)

        # tf.scan over time axis
        rest_t = tf.transpose(rest, [1, 0, 2])  # (T-1, B, N)
        alphas = tf.scan(step, rest_t, initializer=first)
        last = tf.concat([tf.expand_dims(first, 0), alphas], axis=0)
        # gather alpha at seq_len-1 per batch row
        idx = tf.stack([seq_len - 1, tf.range(tf.shape(potentials)[0])], axis=1)
        last_per_row = tf.gather_nd(last, idx)
        return tf.reduce_logsumexp(last_per_row, axis=-1)

    def _seq_score(self, potentials, tag_ids, seq_len):
        # Sum of unary + transition scores along the gold sequence.
        B = tf.shape(potentials)[0]
        T = tf.shape(potentials)[1]
        mask = tf.sequence_mask(seq_len, T, dtype=potentials.dtype)
        # Unary
        gather_idx = tf.stack(
            [
                tf.repeat(tf.range(B), T),
                tf.tile(tf.range(T), [B]),
                tf.reshape(tag_ids, [-1]),
            ],
            axis=1,
        )
        unary = tf.reshape(tf.gather_nd(potentials, gather_idx), [B, T])
        unary_score = tf.reduce_sum(unary * mask, axis=-1)
        # Transitions
        prev = tag_ids[:, :-1]
        nxt = tag_ids[:, 1:]
        trans = tf.gather_nd(
            self.transitions, tf.stack([prev, nxt], axis=-1)
        )
        trans_mask = mask[:, 1:] * mask[:, :-1]
        trans_score = tf.reduce_sum(trans * trans_mask, axis=-1)
        return unary_score + trans_score

    def _viterbi_decode(self, potentials, seq_len):
        # potentials: (B, T, N)
        B = tf.shape(potentials)[0]
        T = tf.shape(potentials)[1]
        first = potentials[:, 0, :]  # (B, N)

        def step(state, cur):
            prev_score, prev_back = state  # (B, N), (B, N)
            broadcast = tf.expand_dims(prev_score, 2) + self.transitions  # (B, N, N)
            best = tf.reduce_max(broadcast, axis=1) + cur  # (B, N)
            back = tf.argmax(broadcast, axis=1, output_type=tf.int32)  # (B, N)
            return best, back

        rest_t = tf.transpose(potentials[:, 1:, :], [1, 0, 2])  # (T-1, B, N)
        init = (first, tf.zeros_like(first, dtype=tf.int32))
        scores, backs = tf.scan(step, rest_t, initializer=init)
        # scores: (T-1, B, N), backs: (T-1, B, N)
        last_score = tf.concat([tf.expand_dims(first, 0), scores], axis=0)
        last_idx = tf.stack([seq_len - 1, tf.range(B)], axis=1)
        final = tf.gather_nd(last_score, last_idx)  # (B, N)
        last_tag = tf.argmax(final, axis=-1, output_type=tf.int32)  # (B,)

        # Backtrack
        backs_padded = tf.concat([tf.zeros_like(backs[:1]), backs], axis=0)  # (T, B, N)
        # Simple python-side backtrack using tf.while_loop
        def body(t, tags, cur):
            t_idx = t
            back_t = backs_padded[t_idx]  # (B, N)
            cur = tf.gather_nd(back_t, tf.stack([tf.range(B), cur], axis=1))
            tags = tags.write(t_idx - 1, cur)
            return t - 1, tags, cur

        tags_ta = tf.TensorArray(dtype=tf.int32, size=T)
        tags_ta = tags_ta.write(T - 1, last_tag)
        t0 = T - 1
        _, tags_ta, _ = tf.while_loop(
            lambda t, *_: t > 0, body, [t0, tags_ta, last_tag]
        )
        decoded = tf.transpose(tags_ta.stack(), [1, 0])  # (B, T)
        return decoded

    # ------------------------------------------------------------------
    # Loss + accuracy hooks used by build_bilstm_crf in Train.py
    # ------------------------------------------------------------------
    def loss(self, y_true, y_pred_unused):
        # y_pred_unused is the Viterbi decode; we use cached potentials.
        potentials = self._last_potentials
        mask = self._last_mask
        seq_len = self._sequence_lengths(potentials, mask)
        y_true = tf.cast(tf.squeeze(y_true, axis=-1) if len(y_true.shape) == 3 else y_true, tf.int32)
        log_norm = self._log_norm(potentials, seq_len)
        seq_score = self._seq_score(potentials, y_true, seq_len)
        return tf.reduce_mean(log_norm - seq_score)

    def accuracy(self, y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true, axis=-1) if len(y_true.shape) == 3 else y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        mask = self._last_mask
        if mask is None:
            mask = tf.ones_like(y_true, dtype=tf.bool)
        correct = tf.equal(y_true, y_pred)
        correct = tf.logical_and(correct, mask)
        return tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))
