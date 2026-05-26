import pandas as pd
from collections import Counter
import time
import wandb
import pickle
from wandb.keras import WandbCallback

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
To create Neural Network architectures
'''
class NeuralNetwork(object):
    #Setting properties to be used in Neural Network
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.words = list(set(data["Word"].values))
        self.n_words = len(self.words)
        self.tags = list(set(self.data["Tag"].values))
        self.n_tags = len(self.tags)
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(), s["POS"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    #To endcode data for training
    def Data_Encoding(self):
        labels = [[s[2] for s in sent] for sent in self.sentences]
        sentences = [" ".join([s[0] for s in sent]) for sent in self.sentences]
        word_cnt = Counter(self.data["Word"].values)
        vocabulary = set(w[0] for w in word_cnt.most_common(5000))
        self.max_len = 114
        word2idx = {"PAD": 0, "UNK": 1}
        word2idx.update({w: i for i, w in enumerate(self.words) if w in vocabulary})
        tag2idx = {t: i for i, t in enumerate(self.tags)}

        #Saving word2idx, tag2idx for later use
        with open('../data/word2idx.pkl', 'wb') as f:
            pickle.dump(word2idx, f)
        
        with open('../data/tag2idx.pkl', 'wb') as f:
            pickle.dump(tag2idx, f)

        X = [[word2idx.get(w, word2idx["UNK"]) for w in s.split()] for s in sentences]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=word2idx["PAD"])
        y = [[tag2idx[l_i] for l_i in l] for l in labels]
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=tag2idx["|O\n"])
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        print("Completed till split")
    
    #LSTM Model for training
    def LSTM_NN(self):
        wandb.init(project="GALE_LIME_NER_LSTM_CRF_DISEASE", entity="robofied")
        wandb.config = {"learning_rate": 0.001,
                        "epochs": 20,
                        "batch_size": 32}

        word_input = keras.Input(shape=(self.max_len,))
        model = layers.Embedding(input_dim=self.n_words, output_dim=50, input_length=self.max_len)(word_input)
        model = layers.SpatialDropout1D(0.1)(model)
        model = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = layers.TimeDistributed(layers.Dense(self.n_tags, activation="softmax"))(model)
        model = keras.Model(word_input, out)
        opt = keras.optimizers.Adam(learning_rate = 0.001)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.history = model.fit(self.X_tr, self.y_tr.reshape(*self.y_tr.shape, 1), batch_size=32, epochs=20, validation_split=0.2, verbose=1, callbacks=[WandbCallback()])
        name = '../models/' + 'ckpt' +str(time.time()) + '.h5'
        model.save(name)
        print("Model saved in model directory...")
        return self.history

    #Training Plots for Accuracy and Loss
    def Training_Plots(self):
        history = self.history
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        acc_fig_name = '../figures/' + 'ckpt_acc' +str(time.time()) + '.png'
        plt.savefig(acc_fig_name)
        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        loss_fig_name = '../figures/' + 'ckpt_loss' +str(time.time()) + '.png'
        plt.savefig(loss_fig_name)
