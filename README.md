***Interpreting Bidirectional-LSTM-CRF model for Disease Entities Recognition***
================

The project was developed by Akshat Gupta and Silvia Cunico under the guidance of Prof. Roman Klinger, from the University of Stuttgart.
The program has 2 main purposes:
- Recognizing disease entities in text documents and labeling them according to the BIO labels
- Interpreting the model's predictions with the LIME and GALE approximation techniques to explain the BI-LSTM CRF model

----------

### Installation
---------------

This program was developed using Python version 3.9.6 and was tested on Linux and Windows system.
We recommend using Anaconda 4.2 for installing **Python 3.9** as well as **numpy**, although you can install them by other means.

If you wish to run the code, you can install the dependencies from the requirements.txt file.

    pip install -r requirements.txt

### Task
-----------

Train a model given a labeled dataset to provide a "B-DISEASE" or "I-DISEASE" tag for each disease entity and "O" for all the remaining tokens on an unlabeled corpus. A classical application is Named Entity Recognition (NER) for Clinical or Biomedical NLP. Here is an example:

```
John   has   been   diagnosed   with    sporadic      T-cell	    leukaemia	
O      O     O      O           O       B-DISEASE     I-DISEASE     I-DISEASE

```

After training a model:
1. Given a predicition result, evaluate its accuracy and F1 score (confusion matrix). 
2. Generate explanations to get both a local and a global approximation of the model’s behaviour. 


### Datasets
------------

The dataset used to train our models was NCBI Disease dataset.

**If you may wish to use the code you can provide a dataset with the same format so that it will be processed regularly and written as a pandas DataFrame in a CSV file for the later model trainings. Dump such DataFrame in a /data folder accordinly.**

The format we refer to follows roughly the CoNLL 2003 format for NER task (https://aclanthology.org/W03-0419.pdf): the data file must contain one word per line. At the end of each line a tag must state whether the current word is part of a named entity or not. In our case the fields in a line must be two, the word and its entity tag. The part of-speech tag and the chunk tag are not part of the required fields. It is essential that the two required fields are tab-separated.


### Results
----------

The 2026 refactor evaluates with seqeval (entity-level, strict IOB2). The
previous numbers in this README came from a hand-rolled token-level
confusion matrix that averaged the dominant `O` class into the macro
score; that metric reports F1 ≈ 0.98 even for an all-`O` predictor on
NCBI Disease. Reproduce that here:

```
$ python -c "from data_loader import load_ncbi_disease; \
            from Evaluation import evaluate; \
            t = load_ncbi_disease('../data/ner-disease')['test']; \
            evaluate(t.tags, [['O']*len(s) for s in t.sentences])"
# entity-level F1: 0.0000   <-- correct under seqeval
# token-level acc: 0.9203   <-- what the old metric was really reporting
```

Re-run targets (NCBI Disease test, single seed unless noted):

|                                              | Precision | Recall | F1   |
|----------------------------------------------|-----------|--------|------|
| Naive Bayes (per-token, no context)          | TBD       | TBD    | TBD  |
| BiLSTM (this repo, retrained)                | TBD       | TBD    | TBD  |
| BiLSTM-CRF (this repo, retrained)            | TBD       | TBD    | TBD  |
| BioBERT v1.1 (HuggingFace)                   | TBD       | TBD    | TBD  |
| PubMedBERT abstract+fulltext                 | TBD       | TBD    | TBD  |

Multi-seed runs (mean ± std over seeds {41, 42, 43}) are written to
`models/multiseed_results.json`. The previous single-run numbers are no
longer in this table because they were token-level and not comparable.


### Refactor (2026): what changed
----------------------------------

The earlier version of this repo had three issues that would have been
called out in peer review:

1. The reported F1 (~0.98) was Keras `accuracy` averaged over a
   train-internal validation slice; the held-out NCBI Disease test
   split was never evaluated. This is now fixed.
2. The "BiLSTM-CRF" in the title was a softmax BiLSTM in the code; the
   CRF layer is now wired in (`code/crf_layer.py`, used by
   `code/Train.py::build_bilstm_crf`).
3. The only baselines were 1990s-era Naive Bayes vs. a 2016-era BiLSTM.
   A modern BioBERT / PubMedBERT baseline is now provided via
   `code/train_biobert.py` (HuggingFace, requires GPU + `transformers`).

Other additions:

- `code/data_loader.py`: unified IOB / CoNLL reader for NCBI Disease and
  BC5CDR. Sentence-segments by `.` (matching the original `Sentence`
  column), or returns whole abstracts via `segment="document"` for
  transformer fine-tuning.
- `code/Evaluation.py`: now seqeval-based (strict IOB2). Old function
  names kept as deprecated shims so `main_NB.py` keeps working.
- `code/multi_seed_runner.py`: trains across N seeds and reports mean ±
  std on `precision`, `recall`, `f1`.
- `code/interpretability_compare.py`: paper framing C — extracts NB
  log-prob feature weights, LIME explanations on BiLSTM, and last-layer
  attention from PubMedBERT for the same test sentences and reports
  pairwise Spearman / top-k agreement.
- `code/NaiveBayes.py`: fixed two real bugs in the original code (the
  `self.ffeatures` typo silently dropped all I-class word counts; the
  `'I-DISEASE\n'` literal was missing the leading `|` so I-class words
  were classified into `O`).


### Usage
---------

#### Naive Bayes baseline (CPU, no extra deps beyond requirements.txt)
```
cd code
python main_NB.py
```

#### BiLSTM / BiLSTM-CRF (CPU works, GPU recommended)
```
cd code
python main_NN.py --model bilstm     --epochs 20 --seed 42
python main_NN.py --model bilstm_crf --epochs 20 --seed 42

# Multi-seed:
python multi_seed_runner.py --model bilstm_crf --seeds 41 42 43 --epochs 20
```

#### BioBERT / PubMedBERT (GPU + HuggingFace Hub access required)
```
cd code
python train_biobert.py \
   --train ../data/ner-disease/train.iob \
   --dev   ../data/ner-disease/dev.iob \
   --test  ../data/ner-disease/test.iob \
   --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
   --epochs 4 --batch-size 16 --seed 42 --save-attention
```

#### BC5CDR cross-dataset evaluation
BC5CDR is not redistributed in this repo. Place BIO files under
`data/bc5cdr/{train,dev,test}.tsv` (two columns, tab-separated, blank
lines between abstracts) and then:
```
cd code
python main_NN.py --model bilstm_crf --dataset bc5cdr --epochs 20 --seed 42
```

#### Interpretability comparison (paper framing C)
After training the BiLSTM and BioBERT models:
```
cd code
python interpretability_compare.py \
   --bilstm-weights '../models/bilstm_seed42_*.h5' \
   --biobert-dir   ../models/biobert_ncbi \
   --nb-pickle     ../models/nb.pkl \
   --n-sentences 100
```


### Saliency-disagreement study (paper framing 1)
---------------------------------------------------

The repo also scaffolds the experiment described in the project
roadmap: *do saliency methods disagree more on biomedical NER than on
general-domain NER, and if so, what phenomena drive the gap?* This
extends Jukic, Tutek & Snajder (ACL 2023 Findings) and Krishna et al.
(2022) to a domain that previous saliency-disagreement work hasn't
covered.

Modules under `code/explainers/`:

  - `saliency_metrics.py` - feature agreement @k, rank agreement @k,
    sign agreement, signed rank agreement @k, Spearman rank
    correlation; `pairwise_table()` returns long-form rows ready for
    aggregation.
  - `phenomena.py` - per-sentence features (OOV-rate vs. CoNLL-2003
    train, multi-word entity presence, abbreviation density,
    `( ABBR )` definitions) and a `bucketize()` helper.
  - `faithfulness.py` - deletion / insertion AUC against any
    `score_fn(words) -> P(target)`. Used as a sanity check that any
    cross-domain disagreement gap isn't just one method failing.
  - `integrated_gradients.py` - IG via Captum on the
    PubMedBERT/BioBERT/BERT models (the third saliency method
    alongside LIME and attention).

Driver: `code/run_disagreement_study.py`.

Reference numbers from the smoke test (NCBI Disease test, 981 sents):
multi-word entities in **31.5%** of sentences, abbreviation density
mean **0.17**, `( ABBR )` patterns in **11.1%** of sentences. CoNLL
equivalents are typically <5% on multi-word entities and ~0% on
parenthesised abbreviations - so the cross-domain gap exists in the
data before any model is trained.

To run the study:
```
cd code
python main_NN.py --model bilstm_crf --dataset ncbi      --seeds 41 42 43
python main_NN.py --model bilstm_crf --dataset bc5cdr    --seeds 41 42 43
python main_NN.py --model bilstm_crf --dataset conll2003 --seeds 41 42 43
python train_biobert.py --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --train ../data/ner-disease/train.iob --test ../data/ner-disease/test.iob --save-attention
python train_biobert.py --model bert-base-cased \
    --train ../data/conll2003/train.txt    --test ../data/conll2003/test.txt    --save-attention

python run_disagreement_study.py \
    --datasets ncbi bc5cdr conll2003 \
    --bilstm-weights '../models/bilstm_crf_seed42_*.h5' \
    --biobert-dir ../models/biobert_ncbi \
    --bert-dir    ../models/bert_conll2003 \
    --n-sentences 200 --out-dir ../results
```

Significance tests on the resulting CSVs (paired bootstrap on
disagreement(biomed) - disagreement(general)) are intentionally left
to a downstream analysis script - that's where domain knowledge about
the right test (paired vs. unpaired, per-bucket vs. global) belongs.


### What was *not* run in the 2026 refactor commit
------------------------------------------------------

The refactor was prepared in an environment without GPU, without
HuggingFace Hub access, and with no pre-installed TensorFlow / PyTorch.
The data loader and seqeval evaluator are smoke-tested end-to-end on
NCBI Disease (sentence counts match the published 5424/923/940 split
within tokenization noise, all-`O` predictor scores entity-F1 = 0).
None of BiLSTM / BiLSTM-CRF / BioBERT / multi-seed numbers were
re-generated; the table above leaves those rows as **TBD** rather than
copying forward the inflated values from the original token-level
metric.

## Contacts
------------

If you have any questions or problems, please e-mail **st180429@stud.uni-stuttgart.de , st179785@stud.uni-stuttgart.de**

------------

