''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''

import ast
import pandas as pd
from DataPreperation import Preprocess
from EDA_on_Data import EDA
from Train import NeuralNetwork
from Explainer import explaination_generator
from BioBertEmbeddings import BioBERTEmbedding


def main():
    train = '../data/ner-disease/train.iob'

    preprocess = Preprocess()
    preprocess.text_to_data(filepath=train)
    preprocess.preprocess_data()
    preprocess.create_dataframe()
    preprocess.dump_csv()

    eda = EDA()
    eda.read_csv()
    eda.EDA()

    data = pd.read_csv("../data/dfnew.csv", encoding="latin1").fillna(method="ffill")

    getData = NeuralNetwork(data)
    getData.Data_Encoding()
    print("Data Encoding is Completed...")
    getData.LSTM_NN()
    print("Trainign NN is completed...")
    getData.Training_Plots()
    print("Curves Plotted and Stored in the Directory...")

    # Explainer Word Importance Generation
    df = pd.read_csv('../data/ner-disease/DatasetTrain.csv')
    sentences, temp = [], []

    for i in range(len(df['Word']) - 1):
        if df['Sentence'][i] == df['Sentence'][i + 1]:
            temp.append(df['Word'][i])
        elif df['Sentence'][i] != df['Sentence'][i + 1]:
            sent_temp = ''.join(str(temp))
            sentences.append(sent_temp)
            temp = []

    df_sents = pd.DataFrame(columns=['sentences'])
    df_sents['sentences'] = sentences
    df_sents.to_csv('../data/ner-disease/sentences.csv')

    words_final = []
    for item in df_sents['sentences']:
        refined_sentence = ast.literal_eval(item)
        final_sentece = ' '.join(refined_sentence)
        wi = explaination_generator(final_sentece)
        print(wi)
        words_final.append(wi)

    words_df = pd.DataFrame(columns=['dicts'])
    words_df['dicts'] = words_final
    words_df.to_csv('../data/ner-disease/words_dicts.csv')

    BioBERTEmbedding.Get_BioBertEmbedding()
    print("BioBert Embeddings Generated and Plots are saved in figures directory.")


if __name__ == "__main__":
    main()