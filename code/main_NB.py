''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''

#Import our custom classes for Preprocess, Naive Bayes, and Evaluation

from Preprocess import Preprocess
from NaiveBayes import NaiveBayes
from Evaluation import PrecisionRecallEntityLevel, PrecisionRecall, PrecisionRecallEntityLevelGene


def run_disease_pipeline():
    train = '../data/ner-disease/train.iob'
    test = '../data/ner-disease/test.iob'

    preprocess = Preprocess()
    preprocess.text_to_data(filepath=train)
    X, y = preprocess.preprocess_data()
    preprocess.text_to_data(filepath=test)
    Xtest, y_true = preprocess.preprocess_data()

    nb = NaiveBayes()
    nb.MultinomialNBTrain(X, y)

    y_pred = [nb.MultinomialNBTest(item) for item in Xtest]

    print("Results for Disease Level NER:")
    PrecisionRecallEntityLevel(Xtest, y_pred, y_true)


def run_gene_pipeline():
    train = '../data/ner-gene/train.iob'
    test = '../data/ner-gene/test.iob'

    preprocess = Preprocess()
    preprocess.text_to_data(filepath=train)
    X, y = preprocess.preprocess_data()
    preprocess.text_to_data(filepath=test)
    Xtest, y_true = preprocess.preprocess_data()

    nb = NaiveBayes()
    nb.MultinomialNBTrainGene(X, y)

    y_pred = [nb.MultinomialNBTestGene(item) for item in Xtest]

    print("Results for Gene Level NER:")
    PrecisionRecallEntityLevelGene(Xtest, y_pred, y_true)


if __name__ == "__main__":
    run_disease_pipeline()
    run_gene_pipeline()

# For later use
# ------------------------------------------------------
# st.title("NER for Disease and Gene")
# raw_text = st.text_area("Your Text","Enter Text Here")
# tokens = raw_text.split()
# custom_pred = []

# if (st.button('Submit')):
#     for item in tokens:
#         pred = nb.MultinomialNBTest(item)
#         custom_pred.append(pred)
#     st.text(custom_pred)



