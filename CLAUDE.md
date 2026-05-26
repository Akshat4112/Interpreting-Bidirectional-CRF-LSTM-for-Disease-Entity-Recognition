# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Biomedical Named Entity Recognition (NER) for disease entities, with model interpretability via LIME / GALE approximations. Trained on the NCBI Disease dataset; a parallel gene/protein pipeline reuses the same Naive Bayes baseline.

Two parallel pipelines:
1. **Naive Bayes baseline** (`code/main_NB.py`) — a from-scratch Multinomial NB token classifier in `code/NaiveBayes.py`, evaluated with custom entity-level metrics.
2. **Bi-LSTM neural pipeline** (`code/main_NN.py`) — preprocesses raw IOB into a CSV, trains a Bidirectional-LSTM (embedding + SpatialDropout + BiLSTM + TimeDistributed Dense softmax), generates LIME explanations per word, then computes BioBERT embeddings + t-SNE.

Note: Despite the repo name "Bidirectional-CRF-LSTM", the active model in `code/Train.py` is plain BiLSTM with softmax — no CRF layer. The CRF variant lives in `archive/Train.py` (uses `keras-contrib`'s CRF and is not wired into the current pipeline; the `keras-contrib/` folder at repo root is empty).

## Environment & commands

Python 3.9.6 (per README). Tested on Linux and Windows. Install with:

```
pip install -r requirements.txt
```

**All scripts must be run from inside `code/`** — every path is relative (`../data/...`, `../models/...`, `../figures/...`). Running from repo root will fail.

```
cd code
python main_NB.py        # Naive Bayes pipeline (disease + gene)
python main_NN.py        # Full BiLSTM pipeline: preprocess -> train -> LIME -> BioBERT t-SNE
python Evaluation.py     # Standalone — note: file has no __main__ guard, importing it runs nothing
python Explainer.py      # Standalone interpretability runs (requires trained .h5 + pickles)
```

There is no test suite, linter config, or build step. There is no `__main__` guard anywhere — modules execute their entrypoint as soon as `python <file>.py` runs the script body at module scope.

## Data conventions

- **Input format**: tab-separated IOB files at `data/ner-disease/{train,test,dev,dev-predicted}.iob` and `data/ner-gene/*.iob` (CoNLL 2003 style, two fields: token, tag).
- **Critical: tags carry literal trailing newlines and a leading `|`.** Comparisons are done against string literals exactly as `'|B-DISEASE\n'`, `'|I-DISEASE\n'`, `'|O\n'` (and `|B-PROTEIN\n` / `|I-PROTEIN\n` for the gene pipeline). Any code that strips whitespace from tags will break matching across `NaiveBayes`, `Evaluation`, `Explainer`, and `BioBertEmbeddings`.
- **DataFrame artifact**: `main_NN.py`'s preprocessing writes `data/dfnew.csv` with columns `Sentence, Word, POS, Tag`. POS tags are filled via `nltk.pos_tag` token-by-token; sentence IDs increment on each `.` token. EDA and training both read this CSV.
- **Vocab/label artifacts**: `Train.py` pickles `data/word2idx.pkl` and `data/tag2idx.pkl` (vocab capped at top-5000 most-common words, plus `PAD`=0 and `UNK`=1). `Explainer.py` and any inference must load these same pickles. `max_len = 114` is hard-coded.
- **Model checkpoints**: saved as `models/ckpt<time.time()>.h5`. `Explainer.py` hard-codes a specific checkpoint filename (`ckpt1658660485.8331368.h5`); update it when retraining if you intend to run the explainer afterwards.
- **Figures**: accuracy/loss plots written to `figures/ckpt_acc<ts>.png` and `figures/ckpt_loss<ts>.png`.

## Experiment tracking

`Train.py` calls `wandb.init(project="GALE_LIME_NER_LSTM_CRF_DISEASE", entity="robofied")` and uses `WandbCallback()` during `model.fit`. Either log into a wandb account that has access to that entity, run `wandb offline`, or unset/replace these calls before training.

## Pipeline data flow (BiLSTM)

`main_NN.py` orchestrates this exact sequence — each step writes artifacts the next reads:

1. `DataPreperation.Preprocess` reads `train.iob`, builds the (Sentence, Word, POS, Tag) DataFrame, writes `data/dfnew.csv`.
2. `EDA_on_Data.EDA` reads `dfnew.csv` and prints stats / draws histograms.
3. `Train.NeuralNetwork.Data_Encoding` builds `word2idx`/`tag2idx` from `dfnew.csv`, pads to `max_len=114`, pickles both maps, splits 80/20 with `shuffle=False`.
4. `Train.LSTM_NN` trains for 20 epochs (batch_size=32, Adam lr=0.001, sparse categorical CE), saves `.h5`.
5. `Train.Training_Plots` writes accuracy/loss PNGs.
6. `main_NN.py` then reads `data/ner-disease/DatasetTrain.csv` (different file from `dfnew.csv` — manually curated), reconstructs sentences by walking the `Sentence` column, and calls `Explainer.explaination_generator` per sentence to produce `data/ner-disease/words_dicts.csv`.
7. `BioBertEmbeddings.Get_BioBertEmbedding` loads `nlu.load('biobert pos')` and runs t-SNE on the embeddings (this requires `spark-nlp` / `nlu` + Java to be configured).

## Naive Bayes pipeline notes

`main_NB.py` runs the disease pipeline then the gene pipeline back-to-back. `NaiveBayes.py` is a from-scratch implementation (log-probabilities, add-one smoothing). The label-matching strings differ per pipeline: `|B-DISEASE\n` vs `|B-PROTEIN\n`. Evaluation uses `PrecisionRecallEntityLevel` (disease) / `PrecisionRecallEntityLevelGene` (gene) — entity-level metrics that scan for `B` followed by consecutive `I` tags, not the token-level `PrecisionRecall`.

## Things to be aware of when editing

- Both `Train.py` files (`code/` and `archive/`) exist; only `code/Train.py` is imported by `main_NN.py`. Don't confuse them.
- `DataPreperation.Preprocess` and `Preprocess.Preprocess` are two different classes used by the two pipelines — they share a name but live in separate modules. `main_NN.py` imports the former, `main_NB.py` the latter.
- Tag-string comparisons must preserve the leading `|` and trailing `\n`. When in doubt, print `repr(tag)` rather than `tag`.
- Scripts have side effects at import time (no `if __name__ == "__main__":`), so importing a module from this `code/` directory will execute its top-level code (e.g. `wandb.init` fires on `import Train`).
- Relative paths assume `code/` is the cwd; introducing new entrypoints from the repo root requires either `os.chdir` or rewriting paths.
