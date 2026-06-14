#!/usr/bin/env python3
"""
Generate a complete technical PDF report describing the
"Interpreting Bidirectional LSTM-CRF for Disease Entity Recognition" project.

Pure-Python (fpdf2 + bundled DejaVu Unicode fonts). Run:

    python docs/generate_report.py

Writes: Interpreting_BiLSTM_CRF_Disease_NER_Report.pdf at the repo root.
"""
from __future__ import annotations

import os
from fpdf import FPDF
from fpdf.fonts import FontFace
from fpdf.enums import XPos, YPos

# --------------------------------------------------------------------------
# Paths / palette
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FIG_DIR = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "Interpreting_BiLSTM_CRF_Disease_NER_Report.pdf")

NAVY = (31, 59, 95)        # primary
TEAL = (0, 121, 140)       # accent
LIGHT = (236, 241, 247)    # light panel
CODE_BG = (244, 245, 247)  # code background
GREY = (110, 118, 130)
RULE = (200, 208, 218)
BOX = (223, 234, 246)

ACC_FIG = os.path.join(FIG_DIR, "ckpt_acc1659180753.4390435.png")
LOSS_FIG = os.path.join(FIG_DIR, "ckpt_loss1659180753.802896.png")


# --------------------------------------------------------------------------
# PDF subclass with header/footer + helpers
# --------------------------------------------------------------------------
class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(20, 18, 20)
        self.add_font("DejaVu", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf")
        self.add_font("Mono", "", os.path.join(FONT_DIR, "DejaVuSansMono.ttf"))
        self.add_font("Mono", "B", os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf"))
        self.title_on_page = False
        self.set_title("Interpreting Bidirectional LSTM-CRF for Disease Entity Recognition")
        self.set_author("Akshat Gupta and Silvia Cunico")

    # --- chrome -----------------------------------------------------------
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(-12)
        self.set_font("DejaVu", "I", 7.5)
        self.set_text_color(*GREY)
        self.cell(0, 6, "Interpreting Bi-LSTM-CRF for Disease Entity Recognition",
                  align="L")
        self.cell(0, 6, f"{self.page_no()}", align="R")
        self.set_text_color(0, 0, 0)

    def header(self):
        if not self.title_on_page or self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(*GREY)
        self.set_y(8)
        self.cell(0, 5, "Technical Report  ·  Biomedical NER & Model Interpretability",
                  align="R")
        self.set_text_color(0, 0, 0)
        self.set_y(self.t_margin)

    # --- ensure vertical space --------------------------------------------
    def need(self, h):
        if self.get_y() + h > self.page_break_trigger:
            self.add_page()

    # --- headings ---------------------------------------------------------
    def h1(self, number, title):
        self.add_page()
        self.start_section(f"{number}  {title}")
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        self.set_font("DejaVu", "B", 15)
        self.set_x(self.l_margin)
        self.cell(0, 11, f"  {number}   {title}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def h2(self, number, title):
        self.need(20)
        self.start_section(title, level=1)
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(*NAVY)
        self.cell(0, 8, f"{number}  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        y = self.get_y()
        self.line(self.l_margin, y, self.l_margin + 28, y)
        self.set_text_color(0, 0, 0)
        self.ln(2.5)

    def h3(self, title):
        self.need(14)
        self.set_font("DejaVu", "B", 10.5)
        self.set_text_color(*TEAL)
        self.cell(0, 6.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(0.5)

    # --- body text --------------------------------------------------------
    def body(self, text, size=10, gap=2.2):
        self.set_font("DejaVu", "", size)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5.2, text, align="J",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(gap)

    def bullets(self, items, size=10):
        self.set_font("DejaVu", "", size)
        self.set_text_color(20, 20, 20)
        for it in items:
            self.need(7)
            x0 = self.get_x()
            self.set_text_color(*TEAL)
            self.cell(6, 5.1, "•")
            self.set_text_color(20, 20, 20)
            self.multi_cell(self.w - self.r_margin - self.get_x(), 5.1, it,
                            align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_x(x0)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def numbered(self, items, size=10):
        self.set_font("DejaVu", "", size)
        for i, it in enumerate(items, 1):
            self.need(7)
            x0 = self.get_x()
            self.set_text_color(*TEAL)
            self.set_font("DejaVu", "B", size)
            self.cell(7, 5.1, f"{i}.")
            self.set_font("DejaVu", "", size)
            self.set_text_color(20, 20, 20)
            self.multi_cell(self.w - self.r_margin - self.get_x(), 5.1, it,
                            align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_x(x0)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    # --- code block -------------------------------------------------------
    def code(self, text, size=8):
        lines = text.strip("\n").split("\n")
        self.set_font("Mono", "", size)
        line_h = size * 0.50
        pad = 2.4
        block_h = line_h * len(lines) + 2 * pad
        self.need(block_h + 3)
        x = self.l_margin
        y = self.get_y()
        w = self.w - self.l_margin - self.r_margin
        self.set_fill_color(*CODE_BG)
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.rect(x, y, w, block_h, style="DF")
        # teal accent bar
        self.set_fill_color(*TEAL)
        self.rect(x, y, 1.1, block_h, style="F")
        self.set_xy(x + 4, y + pad)
        self.set_text_color(35, 40, 48)
        for ln_ in lines:
            self.set_x(x + 4)
            self.cell(w - 6, line_h, ln_, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.set_y(y + block_h)
        self.ln(3)

    # --- callout panel ----------------------------------------------------
    def callout(self, title, text):
        self.set_font("DejaVu", "B", 9.5)
        title_h = 6
        self.set_font("DejaVu", "", 9.5)
        # measure
        self.need(24)
        x = self.l_margin
        y = self.get_y()
        w = self.w - self.l_margin - self.r_margin
        # render text into a temp to find height: just draw with fill panel
        self.set_fill_color(*LIGHT)
        self.set_draw_color(*RULE)
        # First compute lines via split_only
        self.set_font("DejaVu", "", 9.5)
        txt_lines = self.multi_cell(w - 10, 4.8, text, align="L",
                                    new_x=XPos.LMARGIN, new_y=YPos.TOP,
                                    dry_run=True, output="LINES")
        h = title_h + 3 + 4.8 * len(txt_lines) + 5
        self.rect(x, y, w, h, style="DF")
        self.set_fill_color(*NAVY)
        self.rect(x, y, w, title_h + 1.5, style="F")
        self.set_xy(x + 4, y + 0.7)
        self.set_text_color(255, 255, 255)
        self.set_font("DejaVu", "B", 9.5)
        self.cell(w - 8, title_h, title)
        self.set_xy(x + 5, y + title_h + 3)
        self.set_text_color(25, 30, 40)
        self.set_font("DejaVu", "", 9.5)
        self.multi_cell(w - 10, 4.8, text, align="L",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.set_y(y + h)
        self.ln(3)

    # --- table ------------------------------------------------------------
    def make_table(self, headers, rows, widths=None, align=None, fsize=8.5):
        self.set_font("DejaVu", "", fsize)
        head_style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=NAVY)
        kwargs = dict(
            borders_layout="HORIZONTAL_LINES",
            headings_style=head_style,
            line_height=5.0,
            cell_fill_color=(247, 249, 252),
            cell_fill_mode="ROWS",
            first_row_as_headings=True,
            text_align=align or "LEFT",
        )
        if widths:
            kwargs["col_widths"] = widths
        self.set_draw_color(*RULE)
        with self.table(**kwargs) as table:
            r = table.row()
            for hcell in headers:
                r.cell(hcell)
            for data in rows:
                r = table.row()
                for c in data:
                    r.cell(str(c))
        self.ln(3)

    # --- vertical flow diagram -------------------------------------------
    def flow(self, steps, bw=140, bh=10):
        self.need(len(steps) * (bh + 6) + 4)
        cx = self.w / 2
        x = cx - bw / 2
        for i, (label, sub) in enumerate(steps):
            if self.get_y() + bh + 8 > self.page_break_trigger:
                self.add_page()
            y = self.get_y()
            self.set_fill_color(*BOX)
            self.set_draw_color(*NAVY)
            self.set_line_width(0.3)
            self.rect(x, y, bw, bh, style="DF")
            self.set_xy(x, y + 0.3)
            self.set_font("DejaVu", "B", 8.6)
            self.set_text_color(*NAVY)
            self.cell(bw, 4.6, label, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_x(x)
            self.set_font("DejaVu", "", 7.2)
            self.set_text_color(70, 78, 90)
            self.cell(bw, 4.0, sub, align="C")
            self.set_text_color(0, 0, 0)
            self.set_y(y + bh)
            if i < len(steps) - 1:
                ay = self.get_y()
                self.set_draw_color(*TEAL)
                self.set_line_width(0.5)
                self.line(cx, ay, cx, ay + 5.4)
                self.line(cx, ay + 5.6, cx - 1.6, ay + 3.6)
                self.line(cx, ay + 5.6, cx + 1.6, ay + 3.6)
                self.set_y(ay + 6)
        self.ln(4)

    def spacer(self, h=2):
        self.ln(h)


# --------------------------------------------------------------------------
# Build document
# --------------------------------------------------------------------------
def build():
    pdf = Report()
    pdf.set_text_color(0, 0, 0)

    # ===================== TITLE PAGE =====================
    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 78, style="F")
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 78, pdf.w, 2.5, style="F")

    pdf.set_xy(0, 20)
    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(200, 215, 232)
    pdf.cell(0, 6, "TECHNICAL REPORT", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_x(0)
    pdf.set_font("DejaVu", "B", 23)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 11,
                   "Interpreting a Bidirectional\nLSTM-CRF Model for\nDisease Entity Recognition",
                   align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(92)
    pdf.set_font("DejaVu", "I", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 7, "Biomedical Named Entity Recognition meets Explainable AI",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)

    # Authorship panel
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("DejaVu", "", 10.5)
    info = [
        ("Authors", "Akshat Gupta  &  Silvia Cunico"),
        ("Supervision", "Prof. Roman Klinger, University of Stuttgart"),
        ("Domain", "Clinical / Biomedical NLP  ·  Sequence Labelling  ·  Interpretability"),
        ("Primary dataset", "NCBI Disease Corpus  (parallel gene/protein pipeline)"),
        ("Code base", "Python 3.9+  ·  TensorFlow/Keras  ·  HuggingFace  ·  seqeval  ·  eli5"),
    ]
    pdf.ln(2)
    box_x = 32
    box_w = pdf.w - 2 * box_x
    pdf.set_x(box_x)
    pdf.set_draw_color(*RULE)
    for k, v in info:
        pdf.set_x(box_x)
        pdf.set_font("DejaVu", "B", 10)
        pdf.set_text_color(*TEAL)
        pdf.cell(34, 6.6, k)
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(35, 35, 35)
        pdf.multi_cell(box_w - 34, 6.6, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Abstract teaser
    pdf.ln(8)
    pdf.set_x(box_x)
    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(*RULE)
    abstract = ("This report documents a research code base that trains and compares "
                "models for recognising disease (and gene/protein) mentions in biomedical "
                "text, and then explains their predictions. It spans a from-scratch Naive "
                "Bayes baseline, a BiLSTM and a custom linear-chain BiLSTM-CRF, and a modern "
                "BioBERT / PubMedBERT transformer baseline, all evaluated with strict "
                "entity-level seqeval metrics. On top of recognition, it provides a suite "
                "of interpretability tools - LIME, Integrated Gradients, attention and a "
                "cross-method saliency-disagreement study - aimed at understanding what "
                "sequence models capture that count-based methods cannot.")
    lines = pdf.multi_cell(box_w - 8, 4.8, abstract, align="J", dry_run=True,
                           output="LINES", new_x=XPos.LMARGIN, new_y=YPos.TOP)
    ah = 4.8 * len(lines) + 12
    y = pdf.get_y()
    pdf.rect(box_x, y, box_w, ah, style="DF")
    pdf.set_xy(box_x + 4, y + 3)
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 5, "ABSTRACT", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(box_x + 4, y + 9)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(35, 35, 35)
    pdf.multi_cell(box_w - 8, 4.8, abstract, align="J")

    pdf.set_y(-30)
    pdf.set_font("DejaVu", "I", 8.5)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, "Generated from repository sources  ·  Document version 1.0",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)

    # ===================== TABLE OF CONTENTS =====================
    pdf.add_page()
    pdf.title_on_page = True
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 11, "Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 30, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.insert_toc_placeholder(render_toc, pages=2)

    # ===================== 1. EXECUTIVE SUMMARY =====================
    pdf.h1("1", "Executive Summary")
    pdf.body(
        "Named Entity Recognition (NER) over biomedical text is the task of locating and "
        "labelling spans that refer to entities of interest - here, diseases (and, in a "
        "parallel pipeline, genes/proteins). This project pursues two coupled goals: (1) "
        "build models that tag each token of a sentence with a BIO label, and (2) interpret "
        "those models so that their decisions can be inspected and trusted, which is "
        "essential in clinical and biomedical settings.")
    pdf.body(
        "The code base is organised as two historical pipelines plus a 2026 refactor that "
        "modernised both evaluation and modelling. The original work (2022) shipped a "
        "from-scratch Multinomial Naive Bayes tagger and a Keras BiLSTM, explained with "
        "LIME and visualised with BioBERT embeddings under t-SNE. The refactor added a real "
        "linear-chain CRF layer, a transformer (BioBERT/PubMedBERT) baseline, strict "
        "entity-level evaluation via seqeval, multi-seed reporting, and a cross-method "
        "interpretability / saliency-disagreement study.")
    pdf.h3("Key takeaways")
    pdf.bullets([
        "Four model families are supported: Naive Bayes, BiLSTM, BiLSTM-CRF, and "
        "BioBERT/PubMedBERT - spanning ~30 years of NER methodology.",
        "Evaluation was corrected from an inflated token-level accuracy (~0.98, dominated "
        "by the majority 'O' class) to strict entity-level seqeval F1, under which an "
        "all-'O' predictor correctly scores F1 = 0.",
        "Interpretability is treated as a first-class deliverable: LIME, Integrated "
        "Gradients, attention, and Naive-Bayes log-probability weights are compared "
        "head-to-head on the same sentences.",
        "A saliency-disagreement study asks whether explanation methods disagree more on "
        "biomedical NER than on general-domain NER, and which linguistic phenomena drive "
        "any gap.",
    ])

    # ===================== 2. INTRODUCTION =====================
    pdf.h1("2", "Introduction and Motivation")
    pdf.h2("2.1", "The recognition task")
    pdf.body(
        "Given a tokenised sentence, the model assigns each token one of three labels in "
        "the BIO (also called IOB2) scheme: B-DISEASE marks the first token of a disease "
        "mention, I-DISEASE marks a subsequent token inside the same mention, and O marks "
        "any token outside an entity. Entity spans are therefore recovered by scanning for "
        "a B tag followed by zero or more consecutive I tags of the same type.")
    pdf.h3("Worked example")
    pdf.code(
        "John   has   been   diagnosed   with   sporadic     T-cell      leukaemia\n"
        "O      O     O      O           O      B-DISEASE    I-DISEASE   I-DISEASE")
    pdf.body(
        "Here \"sporadic T-cell leukaemia\" is a single three-token disease entity. The "
        "challenge is that disease mentions are frequently multi-word, contain "
        "abbreviations and parenthesised acronyms, and use vocabulary that is far from "
        "everyday English - exactly the properties this project later quantifies.")
    pdf.h2("2.2", "Why interpretability")
    pdf.body(
        "A tagger that is accurate but opaque is hard to deploy in biomedical practice. "
        "The project therefore explains predictions both locally (why did the model tag "
        "this token this way?) and globally (which words, across the corpus, push the model "
        "toward a disease label?). The two requested approximation techniques are LIME "
        "(local, perturbation-based) and GALE (a global aggregation of local explanations).")
    pdf.callout(
        "Central research question",
        "What do sequence models \"see\" in NER that a count-based Naive Bayes classifier "
        "cannot? The Naive Bayes baseline ignores word order, part-of-speech, and context; "
        "the LSTM adds sequential memory and the CRF adds label-transition structure. The "
        "interpretability layer is the instrument used to answer this question concretely.")

    # ===================== 3. ARCHITECTURE =====================
    pdf.h1("3", "System Architecture Overview")
    pdf.body(
        "The repository contains two parallel, self-contained pipelines plus a set of "
        "modern add-ons. Every script is designed to be run from inside the code/ "
        "directory because all paths are relative to it (../data, ../models, ../figures).")
    pdf.h2("3.1", "The two original pipelines")
    pdf.bullets([
        "Naive Bayes baseline - entry point code/main_NB.py, model in code/NaiveBayes.py. "
        "Runs the disease corpus and then the gene/protein corpus back-to-back, scored "
        "with custom entity-level metrics.",
        "Neural pipeline - entry point code/main_NN.py. Originally: preprocess raw IOB into "
        "a CSV, run exploratory data analysis, train a BiLSTM, generate LIME explanations "
        "per word, then compute BioBERT embeddings and t-SNE.",
    ])
    pdf.h2("3.2", "Repository layout")
    pdf.make_table(
        ["Path", "Role"],
        [
            ["code/main_NB.py", "Naive Bayes pipeline driver (disease + gene)"],
            ["code/main_NN.py", "Neural pipeline CLI (BiLSTM / BiLSTM-CRF, NCBI / BC5CDR)"],
            ["code/NaiveBayes.py", "From-scratch Multinomial Naive Bayes tagger"],
            ["code/Preprocess.py / DataPreperation.py", "IOB readers; CSV + POS feature builder"],
            ["code/data_loader.py", "Unified IOB/CoNLL reader (NCBI, BC5CDR, CoNLL-2003)"],
            ["code/Train.py", "BiLSTM and BiLSTM-CRF construction, training, evaluation"],
            ["code/crf_layer.py", "Self-contained linear-chain CRF Keras layer"],
            ["code/Evaluation.py", "seqeval entity-level metrics (+ legacy shims)"],
            ["code/Explainer.py", "LIME explanations over the BiLSTM (via eli5)"],
            ["code/BioBertEmbeddings.py", "BioBERT embeddings + t-SNE visualisation"],
            ["code/train_biobert.py", "BioBERT / PubMedBERT fine-tuning (HuggingFace)"],
            ["code/multi_seed_runner.py", "Train across seeds; report mean +/- std"],
            ["code/interpretability_compare.py", "NB vs LIME vs attention agreement"],
            ["code/run_disagreement_study.py", "Saliency-disagreement experiment driver"],
            ["code/explainers/", "saliency_metrics, phenomena, faithfulness, IG modules"],
            ["data/, models/, figures/", "Corpora, checkpoints (.h5), training plots"],
            ["archive/", "Earlier keras-contrib CRF prototype (not wired in)"],
        ],
        widths=(42, 58), fsize=8.3,
    )
    pdf.h2("3.3", "Neural pipeline data flow")
    pdf.body(
        "main_NN.py orchestrates an exact sequence in which each step writes an artifact "
        "that the next step consumes:")
    pdf.flow([
        ("Raw IOB corpus", "data/ner-disease/train.iob  (token  \\t  tag)"),
        ("data_loader / DataPreperation", "normalise tags, segment sentences on '.', build DataFrame"),
        ("Encoder.fit / Data_Encoding", "vocab = top-5000 + PAD + UNK; pad to max_len = 114; pickle maps"),
        ("Train (BiLSTM or BiLSTM-CRF)", "Embedding + SpatialDropout + BiLSTM + Dense/CRF"),
        ("seqeval evaluation on test.iob", "strict IOB2 precision / recall / F1; save .h5 checkpoint"),
        ("Explainer (LIME) per sentence", "eli5 TextExplainer -> per-word importance dictionaries"),
        ("BioBERT embeddings + t-SNE", "2-D projection coloured by entity label"),
    ])

    # ===================== 4. DATASETS =====================
    pdf.h1("4", "Datasets and Data Format")
    pdf.h2("4.1", "Corpora")
    pdf.body(
        "The disease models are trained on the NCBI Disease corpus; a parallel pipeline "
        "reuses the same machinery on a gene/protein corpus (PROTEIN labels). The refactor "
        "also adds loaders for BC5CDR (cross-dataset biomedical evaluation) and CoNLL-2003 "
        "(the canonical general-domain comparison corpus), neither of which is "
        "redistributed in the repository.")
    pdf.h2("4.2", "File format and a critical quirk")
    pdf.body(
        "Input files follow the CoNLL-2003 convention loosely: one token per line, with the "
        "token and its tag tab-separated. The NCBI files in this repository use a "
        "non-standard tag encoding - tags carry a leading pipe and a trailing newline, e.g. "
        "the literal strings '|B-DISEASE\\n', '|I-DISEASE\\n', '|O\\n' (and the PROTEIN "
        "equivalents for genes).")
    pdf.callout(
        "Why the quirk matters",
        "The original Naive Bayes and explainer code compares tags against these exact "
        "string literals. Any code that strips whitespace or the leading pipe silently "
        "breaks label matching. The refactor's data_loader._clean_tag normalises every tag "
        "to canonical B-DISEASE / I-DISEASE / O form so that seqeval and HuggingFace "
        "tokenizers can consume it safely, while the legacy NB path keeps the literal "
        "strings it was written against.")
    pdf.h2("4.3", "Corpus statistics")
    pdf.body(
        "Counts below are computed by the repository's own loader (sentences segmented on "
        "the '.' token, the same convention used throughout the pipeline). Note that "
        "sentence-level segmentation yields counts close to, but not identical to, the "
        "published NCBI split (5424 / 923 / 940) because of tokenisation and "
        "segmentation differences.")
    pdf.h3("NCBI Disease")
    pdf.make_table(
        ["Split", "Sentences", "Tokens", "Disease spans (B-)", "B-DISEASE", "I-DISEASE", "O"],
        [
            ["train", "5,915", "128,721", "5,148", "5,148", "4,836", "118,737"],
            ["dev",   "962",   "22,646",  "791",   "791",   "830",   "21,025"],
            ["test",  "981",   "23,175",  "961",   "961",   "866",   "21,348"],
        ],
        widths=(13, 17, 16, 22, 17, 16, 16),
        align="CENTER", fsize=8.0,
    )
    pdf.h3("Gene / Protein")
    pdf.make_table(
        ["Split", "Sentences", "Tokens", "Protein spans (B-)", "B-PROTEIN", "I-PROTEIN", "O"],
        [
            ["train", "9,707", "257,743", "9,700", "9,700", "8,128", "239,915"],
            ["dev",   "5,565", "158,422", "8,465", "8,465", "6,889", "143,068"],
            ["test",  "5,095", "140,006", "6,290", "6,290", "4,801", "128,915"],
        ],
        widths=(13, 17, 16, 22, 17, 16, 16),
        align="CENTER", fsize=8.0,
    )
    pdf.body(
        "The heavy class imbalance is the headline fact: on the NCBI test split, roughly "
        "92% of tokens are 'O'. This is precisely why a token-level accuracy metric is "
        "misleading - a degenerate model that never predicts an entity already scores about "
        "0.92 - and why the project moved to entity-level scoring.")

    # ===================== 5. METHODOLOGY =====================
    pdf.h1("5", "Models and Methodology")

    pdf.h2("5.1", "Naive Bayes baseline (from scratch)")
    pdf.body(
        "NaiveBayes.py implements a Multinomial Naive Bayes token classifier without "
        "external ML libraries. Training counts how often each word occurs under each class "
        "(B, I, O), converts counts to add-one (Laplace) smoothed log-probabilities, and "
        "stores class log-priors. Prediction sums the relevant log-probabilities per class "
        "and returns the arg-max label.")
    pdf.code(
        "# Smoothed class-conditional log-probability (add-one smoothing)\n"
        "features['B'][word] = log((count + 1) / (count_B + doc_count))\n\n"
        "# Prediction: start from log-prior, add evidence, take arg-max\n"
        "p_B = priori_B + sum(features['B'][w] for w in tokens if w in features['B'])\n"
        "label = argmax(p_B, p_I, p_O)")
    pdf.body(
        "By construction this model is bag-of-words at the token level: it has no notion of "
        "word order, part-of-speech, or surrounding context. The project's design document "
        "names these as the three core limitations the neural models are meant to overcome.")
    pdf.callout(
        "Two real bugs fixed in the refactor",
        "(1) A 'self.ffeatures' typo silently dropped all I-class word counts. (2) The "
        "'I-DISEASE' literal was missing its leading pipe, so every I-class word was "
        "misrouted into the O class. Both bugs deflated the baseline; both are corrected.")

    pdf.h2("5.2", "BiLSTM")
    pdf.body(
        "The neural tagger is a word-embedding Bidirectional LSTM with a time-distributed "
        "softmax classifier, built in Train.py::build_bilstm. The bidirectional recurrence "
        "lets each token's representation depend on both left and right context, addressing "
        "the no-sequence-information limitation of Naive Bayes.")
    pdf.make_table(
        ["Layer", "Configuration"],
        [
            ["Input", "integer token IDs, length max_len = 114"],
            ["Embedding", "output_dim = 50, mask_zero = True (PAD = 0)"],
            ["SpatialDropout1D", "rate = 0.1"],
            ["Bidirectional LSTM", "100 units, return_sequences = True, recurrent_dropout = 0.1"],
            ["TimeDistributed Dense", "n_tags units, softmax activation"],
            ["Optimiser / loss", "Adam (lr = 1e-3), sparse categorical cross-entropy"],
            ["Training", "20 epochs, batch_size = 32, validation_split = 0.1"],
        ],
        widths=(42, 58), fsize=8.6,
    )
    pdf.body(
        "Vocabulary is capped at the 5,000 most common training words plus PAD and UNK; "
        "out-of-vocabulary words map to UNK. The (word->index) and (tag->index) maps are "
        "pickled to data/word2idx.pkl and data/tag2idx.pkl so that the explainer and any "
        "inference step reuse exactly the same encoding.")

    pdf.h2("5.3", "BiLSTM-CRF")
    pdf.body(
        "Despite the repository name, the original active model was a plain softmax BiLSTM. "
        "The refactor adds a genuine linear-chain Conditional Random Field as the output "
        "layer (crf_layer.py), implemented from scratch so it depends on neither "
        "keras-contrib nor tensorflow-addons (both unmaintained as of 2025).")
    pdf.body(
        "The CRF replaces the per-token softmax with a structured output that learns "
        "label-to-label transition scores. This directly models constraints such as "
        "\"I-DISEASE may only follow B-DISEASE or I-DISEASE\", giving the contextual "
        "label coherence that a token-independent softmax lacks.")
    pdf.bullets([
        "Emission scores come from the TimeDistributed Dense layer (shape batch x len x "
        "tags); the CRF adds a learned transitions matrix of shape tags x tags.",
        "Training loss is the negative log-likelihood -log P(y | x), where the partition "
        "function log Z is computed with the forward algorithm (implemented with tf.scan "
        "and a log-sum-exp recurrence).",
        "Inference uses Viterbi decoding (a max-plus recurrence with back-pointers, "
        "back-tracked through a tf.while_loop) to return the single best label sequence.",
        "Padding is respected through the propagated Keras mask (mask_zero = True on the "
        "embedding), so padded positions do not contribute to loss or accuracy.",
    ])

    pdf.h2("5.4", "BioBERT / PubMedBERT")
    pdf.body(
        "train_biobert.py fine-tunes a HuggingFace token-classification transformer - by "
        "default microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, with "
        "dmis-lab/biobert-v1.1 as an alternative. This supplies the modern baseline that "
        "the original Naive-Bayes-vs-BiLSTM comparison lacked.")
    pdf.bullets([
        "Word-level IOB labels are aligned to sub-word tokens; only the first sub-token of "
        "each word carries the label and continuation sub-tokens convert B- to I- so that "
        "seqeval decoding stays consistent.",
        "Defaults: 4 epochs, batch size 16, learning rate 3e-5, max length 192; metrics "
        "are computed on the held-out test split with the same seqeval evaluator.",
        "With --save-attention, last-layer mean attention over the test set is cached to "
        "test_attention.npy for the interpretability comparison.",
    ])

    pdf.h2("5.5", "Encoding and preprocessing details")
    pdf.body(
        "DataPreperation.Preprocess builds a (Sentence, Word, POS, Tag) DataFrame, filling "
        "POS via nltk.pos_tag token-by-token and incrementing the sentence ID on each '.' "
        "token; it writes data/dfnew.csv, read by both EDA and training. The unified "
        "data_loader.load_iob is the refactor's replacement: it cleans the tag quirk, "
        "supports sentence-level or document-level segmentation, and returns a typed "
        "NERDataset dataclass shared by every modern script.")

    # ===================== 6. EVALUATION =====================
    pdf.h1("6", "Evaluation Methodology")
    pdf.body(
        "Evaluation.py now computes strict entity-level metrics with seqeval under the IOB2 "
        "scheme. An entity prediction counts as correct only if both its type and its exact "
        "span boundaries match the gold annotation - the standard for NER. A token-level "
        "report (which includes the dominant 'O' class) is printed alongside, but only as a "
        "sanity check.")
    pdf.callout(
        "The metric bug that mattered most",
        "The earlier version reported Keras 'accuracy' over a train-internal validation "
        "slice and never touched the official test split. Because ~92% of tokens are 'O', "
        "that token-level number sits near 0.98 even for a model that predicts no entities "
        "at all. Under strict entity-level seqeval, an all-'O' predictor on NCBI Disease "
        "correctly scores F1 = 0.0000 while still showing ~0.92 token accuracy - the two "
        "numbers the old metric was conflating.")
    pdf.body(
        "Legacy functions (PrecisionRecall, PrecisionRecallEntityLevel, "
        "PrecisionRecallEntityLevelGene) are retained as deprecated shims that forward to "
        "the new evaluator after stripping the tag quirk, so the Naive Bayes driver keeps "
        "working unchanged. multi_seed_runner.py then trains across seeds {41, 42, 43} and "
        "reports mean +/- standard deviation on precision, recall and F1 - the level of "
        "rigour reviewers expect, replacing single-run point estimates.")

    # ===================== 7. INTERPRETABILITY =====================
    pdf.h1("7", "Interpretability")
    pdf.body(
        "Interpretability is the project's distinguishing contribution. Four attribution "
        "signals are produced over the same sentences and compared directly.")
    pdf.h2("7.1", "LIME on the BiLSTM")
    pdf.body(
        "Explainer.py wraps the trained BiLSTM in an eli5 TextExplainer. A "
        "MaskingTextSampler perturbs the input by replacing tokens with UNK; LIME fits a "
        "local linear surrogate to the model's behaviour around each sentence and reports "
        "per-word importance weights for each target tag (B, I, O). The explainer is "
        "position-dependent, so it explains the prediction at a specific token index.")
    pdf.h2("7.2", "GALE - global aggregation")
    pdf.body(
        "Local LIME explanations are aggregated across the corpus to approximate a global "
        "view (GALE) of which words consistently push the model toward each label - moving "
        "from \"why this token here\" to \"what the model has learned in general\".")
    pdf.h2("7.3", "BioBERT embeddings and t-SNE")
    pdf.body(
        "BioBertEmbeddings.py loads a BioBERT pipeline (via the nlu / spark-nlp stack), "
        "embeds sampled B / I / O tokens, and projects them to two dimensions with t-SNE, "
        "colouring points by their named-entity label. Well-separated colour clusters "
        "indicate that contextual biomedical embeddings already encode the entity "
        "distinction geometrically, before any task-specific layer is trained.")
    pdf.h2("7.4", "Cross-method comparison")
    pdf.body(
        "interpretability_compare.py extracts three importance signals for a fixed set of "
        "test sentences - Naive Bayes class-conditional log-probabilities, LIME weights on "
        "the BiLSTM, and last-layer attention from PubMedBERT - and measures how much they "
        "agree, using Spearman rank correlation and top-k overlap. This reframes the work "
        "from a plain benchmark into an interpretability study.")
    pdf.h2("7.5", "Integrated Gradients")
    pdf.body(
        "explainers/integrated_gradients.py adds a third, gradient-based saliency method "
        "via Captum's LayerIntegratedGradients against the transformer's input embeddings, "
        "attributing the predicted-class logit back to each token and mapping sub-word "
        "attributions back to whole words. LIME, attention, and IG together form the three "
        "methods compared in the disagreement study.")

    # ===================== 8. DISAGREEMENT STUDY =====================
    pdf.h1("8", "The Saliency-Disagreement Study")
    pdf.body(
        "The repository scaffolds a focused research experiment that extends prior "
        "disagreement work (Krishna et al. 2022; Jukic, Tutek & Snajder, ACL 2023 Findings) "
        "into a domain those studies did not cover.")
    pdf.callout(
        "Study question",
        "Do saliency methods disagree more on biomedical NER than on general-domain NER, "
        "and if so, which linguistic phenomena drive the gap?")
    pdf.h2("8.1", "Supporting modules")
    pdf.make_table(
        ["Module", "What it provides"],
        [
            ["explainers/saliency_metrics.py",
             "feature/rank/sign/signed-rank agreement @k and Spearman correlation; "
             "pairwise_table() emits long-form rows for aggregation"],
            ["explainers/phenomena.py",
             "per-sentence features: OOV rate vs. CoNLL-2003, multi-word entities, "
             "abbreviation density, parenthesised acronyms; plus bucketize()"],
            ["explainers/faithfulness.py",
             "deletion / insertion AUC against any score_fn - a sanity check that a "
             "disagreement gap is not just one method failing"],
            ["run_disagreement_study.py",
             "driver across NCBI, BC5CDR and CoNLL-2003 x {LIME, IG, attention}; writes "
             "long-form and summary CSVs"],
        ],
        widths=(46, 54), fsize=8.4,
    )
    pdf.h2("8.2", "Reference numbers (smoke test)")
    pdf.body(
        "On the NCBI Disease test set (981 sentences), the phenomena module reports "
        "multi-word entities in 31.5% of sentences, mean abbreviation density 0.17, and "
        "'( ABBR )' acronym patterns in 11.1% of sentences. The CoNLL-2003 equivalents are "
        "typically below 5% for multi-word entities and near 0% for parenthesised "
        "abbreviations - so the cross-domain gap exists in the data before any model is "
        "trained, which is exactly the premise the study sets out to test on explanations.")

    # ===================== 9. REFACTOR =====================
    pdf.h1("9", "The 2026 Refactor: What Changed and Why")
    pdf.body(
        "Three issues in the original code would have been flagged in peer review; the "
        "refactor addresses each:")
    pdf.numbered([
        "Reported F1 (~0.98) was Keras accuracy over a train-internal validation slice; "
        "the held-out NCBI test split was never evaluated. Now fixed with seqeval on the "
        "real test set.",
        "The \"BiLSTM-CRF\" named in the title was a softmax BiLSTM in the code. A real CRF "
        "layer is now wired in via crf_layer.py and Train.build_bilstm_crf.",
        "The only baselines were 1990s-era Naive Bayes vs. a 2016-era BiLSTM. A modern "
        "BioBERT / PubMedBERT baseline is now provided via train_biobert.py.",
    ])
    pdf.h3("Additional improvements")
    pdf.bullets([
        "data_loader.py: one reader for NCBI Disease, BC5CDR and CoNLL-2003, with "
        "sentence- or document-level segmentation and IOB1->IOB2 conversion.",
        "Evaluation.py: seqeval-based, with deprecated shims so main_NB.py still runs.",
        "multi_seed_runner.py: mean +/- std across seeds on every metric.",
        "interpretability_compare.py and the explainers/ package: the analytical core.",
        "NaiveBayes.py: the two correctness bugs described in Section 5.1 fixed.",
    ])
    pdf.callout(
        "Honest reporting of what was NOT run",
        "The refactor was prepared without GPU, without HuggingFace Hub access, and with no "
        "pre-installed TensorFlow/PyTorch. The data loader and seqeval evaluator are "
        "smoke-tested end-to-end (sentence counts match the published split within "
        "tokenisation noise; the all-'O' predictor scores entity-F1 = 0). None of the "
        "BiLSTM / BiLSTM-CRF / BioBERT / multi-seed numbers were regenerated, so the results "
        "table deliberately leaves those rows as TBD rather than carrying forward the "
        "inflated token-level values.")

    # ===================== 10. RESULTS =====================
    pdf.h1("10", "Results")
    pdf.h2("10.1", "Re-run targets")
    pdf.body(
        "The results table is intentionally a set of targets to be filled by a GPU re-run. "
        "Entity-level (strict IOB2) precision, recall and F1 on the NCBI Disease test set "
        "are the reporting standard.")
    pdf.make_table(
        ["Model", "Precision", "Recall", "F1"],
        [
            ["Naive Bayes (per-token, no context)", "TBD", "TBD", "TBD"],
            ["BiLSTM (this repo, retrained)", "TBD", "TBD", "TBD"],
            ["BiLSTM-CRF (this repo, retrained)", "TBD", "TBD", "TBD"],
            ["BioBERT v1.1 (HuggingFace)", "TBD", "TBD", "TBD"],
            ["PubMedBERT abstract+fulltext", "TBD", "TBD", "TBD"],
        ],
        widths=(52, 16, 16, 16), align=("LEFT", "CENTER", "CENTER", "CENTER"),
        fsize=9,
    )
    pdf.h2("10.2", "Historical training curves (legacy BiLSTM)")
    pdf.body(
        "The figures below are the accuracy and loss curves saved by an original 20-epoch "
        "BiLSTM run. They illustrate the metric trap of Section 6: training accuracy climbs "
        "toward 1.0 and validation accuracy plateaus near 0.99 - numbers that look "
        "excellent only because they are dominated by the majority 'O' class. The loss "
        "curves also show the validation loss flattening and slightly rising after a few "
        "epochs, the usual sign of mild over-fitting.")
    if os.path.exists(ACC_FIG) and os.path.exists(LOSS_FIG):
        pdf.need(75)
        img_w = (pdf.w - pdf.l_margin - pdf.r_margin - 6) / 2
        y0 = pdf.get_y()
        pdf.image(ACC_FIG, x=pdf.l_margin, y=y0, w=img_w)
        pdf.image(LOSS_FIG, x=pdf.l_margin + img_w + 6, y=y0, w=img_w)
        pdf.set_y(y0 + img_w * 0.75 + 1)
        pdf.set_font("DejaVu", "I", 8)
        pdf.set_text_color(*GREY)
        pdf.cell(img_w, 5, "Figure 1a. Training vs. validation accuracy.", align="C")
        pdf.cell(6, 5, "")
        pdf.cell(img_w, 5, "Figure 1b. Training vs. validation loss.", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
    pdf.body(
        "These plots are retained as an artifact and a cautionary example; the authoritative "
        "comparison is the entity-level table once regenerated.")

    # ===================== 11. HOW TO RUN =====================
    pdf.h1("11", "How to Run and Reproduce")
    pdf.body("All commands assume the working directory is code/.")
    pdf.h3("Naive Bayes baseline (CPU)")
    pdf.code("cd code\npython main_NB.py")
    pdf.h3("BiLSTM / BiLSTM-CRF")
    pdf.code(
        "python main_NN.py --model bilstm     --epochs 20 --seed 42\n"
        "python main_NN.py --model bilstm_crf --epochs 20 --seed 42\n\n"
        "# multi-seed mean +/- std\n"
        "python multi_seed_runner.py --model bilstm_crf --seeds 41 42 43 --epochs 20")
    pdf.h3("BioBERT / PubMedBERT (GPU + HuggingFace)")
    pdf.code(
        "python train_biobert.py \\\n"
        "  --train ../data/ner-disease/train.iob \\\n"
        "  --dev   ../data/ner-disease/dev.iob \\\n"
        "  --test  ../data/ner-disease/test.iob \\\n"
        "  --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \\\n"
        "  --epochs 4 --batch-size 16 --seed 42 --save-attention")
    pdf.h3("Interpretability comparison and disagreement study")
    pdf.code(
        "python interpretability_compare.py \\\n"
        "  --bilstm-weights '../models/bilstm_seed42_*.h5' \\\n"
        "  --biobert-dir ../models/biobert_ncbi --n-sentences 100\n\n"
        "python run_disagreement_study.py \\\n"
        "  --datasets ncbi bc5cdr conll2003 \\\n"
        "  --bilstm-weights '../models/bilstm_crf_seed42_*.h5' \\\n"
        "  --biobert-dir ../models/biobert_ncbi --bert-dir ../models/bert_conll2003 \\\n"
        "  --n-sentences 200 --out-dir ../results")
    pdf.callout(
        "Operational notes",
        "Run every script from inside code/ (paths are relative). The original scripts have "
        "side effects at import time and the explainer hard-codes a checkpoint filename "
        "(update it after retraining). Train.py also calls wandb.init; log in, run "
        "'wandb offline', or remove the call before training. BC5CDR and CoNLL-2003 must be "
        "placed locally as they are not redistributed.")

    # ===================== 12. ENVIRONMENT =====================
    pdf.h1("12", "Environment and Dependencies")
    pdf.body(
        "Developed and tested with Python 3.9.6 on Linux and Windows; install with "
        "pip install -r requirements.txt. The stack splits along framework lines so the TF "
        "and PyTorch dependencies stay independent.")
    pdf.make_table(
        ["Area", "Key packages"],
        [
            ["Neural (Keras path)", "tensorflow 2.9, keras 2.9"],
            ["Transformers", "torch, transformers, datasets (BioBERT/PubMedBERT)"],
            ["Evaluation", "seqeval (strict IOB2), scikit-learn"],
            ["Interpretability", "lime, eli5, captum, nlu / spark-nlp (BioBERT t-SNE)"],
            ["Data / NLP", "pandas, numpy, nltk (POS tagging)"],
            ["Viz / tracking", "matplotlib, seaborn, wandb"],
        ],
        widths=(34, 66), fsize=9,
    )
    pdf.body(
        "Experiment tracking uses Weights & Biases (project GALE_LIME_NER_LSTM_CRF_DISEASE, "
        "entity robofied). The BioBERT t-SNE step additionally requires a configured "
        "spark-nlp / nlu environment with Java.")

    # ===================== 13. LIMITATIONS =====================
    pdf.h1("13", "Limitations and Future Work")
    pdf.h2("13.1", "Known limitations")
    pdf.bullets([
        "Headline benchmark numbers are not yet regenerated under the corrected metric; the "
        "results table is TBD pending a GPU run.",
        "The BiLSTM vocabulary is capped at 5,000 words, so rare biomedical terms collapse "
        "to UNK - a real constraint given the high OOV rate of the domain.",
        "max_len = 114 is hard-coded; longer abstracts are truncated in sentence mode.",
        "Original scripts execute at import time and the LIME explainer hard-codes a "
        "checkpoint name, so retraining requires manual edits.",
        "The legacy and modern tag conventions coexist (literal '|TAG\\n' vs. canonical "
        "form), a foot-gun for anyone editing the Naive Bayes path.",
    ])
    pdf.h2("13.2", "Future work")
    pdf.bullets([
        "Regenerate the full entity-level results table across all four model families with "
        "multi-seed mean +/- std.",
        "Complete the disagreement study: run all three saliency methods across the three "
        "datasets and add paired-bootstrap significance tests.",
        "Replace the capped-vocabulary embedding with sub-word or contextual embeddings to "
        "reduce the UNK problem in the BiLSTM path.",
        "Add a proper __main__ guard and configuration file to remove import-time side "
        "effects and hard-coded checkpoint names.",
    ])

    # ===================== 14. CONCLUSION =====================
    pdf.h1("14", "Conclusion")
    pdf.body(
        "This project is best understood not as a single model but as a comparative study "
        "of disease-entity recognition across four eras of NER methodology, unified by a "
        "shared evaluation protocol and, crucially, by an interpretability layer that asks "
        "what each model has actually learned. The from-scratch Naive Bayes baseline "
        "establishes a context-free floor; the BiLSTM adds sequential memory; the CRF adds "
        "structured label coherence; and BioBERT/PubMedBERT brings domain-pretrained "
        "contextual representations.")
    pdf.body(
        "The 2026 refactor turned an over-optimistic demonstration into a defensible "
        "experimental scaffold: strict entity-level metrics, a real CRF, a modern "
        "transformer baseline, multi-seed reporting, and a cross-method "
        "saliency-disagreement framework. With the benchmark re-run completed, the code base "
        "is positioned to answer its central question - what sequence and transformer models "
        "see in biomedical NER that count-based methods cannot - with evidence rather than "
        "with an inflated accuracy score.")
    pdf.ln(4)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("DejaVu", "I", 9)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(
        0, 5,
        "Prepared as project documentation for \"Interpreting Bidirectional LSTM-CRF for "
        "Disease Entity Recognition\" by Akshat Gupta and Silvia Cunico, University of "
        "Stuttgart. Statistics and architecture details in this report were extracted "
        "directly from the repository sources.",
        align="C")
    pdf.set_text_color(0, 0, 0)

    pdf.output(OUT)
    return OUT


def render_toc(pdf, outline):
    pdf.set_font("DejaVu", "", 10.5)
    epw = pdf.w - pdf.l_margin - pdf.r_margin
    for section in outline:
        if section.level == 0:
            pdf.ln(1)
            pdf.set_font("DejaVu", "B", 10.5)
            pdf.set_text_color(*NAVY)
            indent = 0
        else:
            pdf.set_font("DejaVu", "", 9.8)
            pdf.set_text_color(60, 66, 76)
            indent = 8
        label = section.name
        page = str(section.page_number)
        pdf.set_x(pdf.l_margin + indent)
        # label width
        lbl_w = pdf.get_string_width(label) + 2
        page_w = pdf.get_string_width(page) + 2
        avail = epw - indent - page_w
        pdf.cell(min(lbl_w, avail), 6.4, label)
        # dot leader
        dots_w = avail - min(lbl_w, avail)
        if dots_w > 4:
            pdf.set_text_color(180, 188, 198)
            ndots = max(0, int(dots_w / pdf.get_string_width(".")))
            pdf.cell(dots_w, 6.4, "." * ndots)
        pdf.set_text_color(*NAVY) if section.level == 0 else pdf.set_text_color(60, 66, 76)
        pdf.cell(page_w, 6.4, page, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


if __name__ == "__main__":
    path = build()
    print("Wrote", path, "(", os.path.getsize(path), "bytes )")
