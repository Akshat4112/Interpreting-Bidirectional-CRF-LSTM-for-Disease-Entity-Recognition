''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''

import glob
import os
import pickle
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
from tensorflow.keras.utils import pad_sequences
from tensorflow import keras
import eli5
import pandas as pd
import ast


def _latest_checkpoint(models_dir='../models'):
    candidates = glob.glob(os.path.join(models_dir, 'ckpt*.h5'))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matching ckpt*.h5 found in {models_dir}. "
            "Run main_NN.py (or Train.py) first."
        )
    return max(candidates, key=os.path.getmtime)

'''
To Create NER Explainer using LIME
'''
class NERExplainerGenerator(object):
    #Loading model, word2idx, tag2idx, and idx2tag
    def __init__(self, model, word2idx, tag2idx, max_len):
        self.model = model
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k,v in tag2idx.items()}
        self.max_len = max_len
        
    # Add Padding and UNK token to make data ready for futher process
    def _preprocess(self, texts):
        X = [[self.word2idx.get(w, self.word2idx["UNK"]) for w in t.split()]
             for t in texts]
        X = pad_sequences(maxlen=self.max_len, sequences=X,
                          padding="post", value=self.word2idx["PAD"])
        return X
    
    # To get prediction from model
    def get_predict_function(self, word_index):
        def predict_func(texts):
            X = self._preprocess(texts)
            p = self.model.predict(X)
            return p[:,word_index,:]
        return predict_func

# To generate explanation for the sentence text
def explaination_generator(text, checkpoint_path=None):

    if checkpoint_path is None:
        checkpoint_path = _latest_checkpoint()
    model = keras.models.load_model(checkpoint_path)
    with open('../data/word2idx.pkl', "rb") as f:
        word2idx = pickle.load(f)

    with open('../data/tag2idx.pkl', "rb") as f1:
        tag2idx = pickle.load(f1)

    #Setting explainer properties
    max_len = 114
    explainer_generator = NERExplainerGenerator(model, word2idx, tag2idx, max_len)
    word_index = 2
    predict_func = explainer_generator.get_predict_function(word_index=word_index)
    sampler = MaskingTextSampler(
        replacement="UNK",
        max_replace=0.7,
        token_pattern=None,
        bow=False
    )

    
    samples, similarity = sampler.sample_near(text, n_samples=4)
    print("Input is: ",samples)
    print("Fitting Explainer...")
    print("----------------------------------------------")
    te = TextExplainer(sampler=sampler,position_dependent=True,random_state=42)
    
    word_importance = {}
    word_importance["B"] = []
    word_importance["I"] = []
    word_importance["O"] = []

    try:
        clf = te.fit(text, predict_func)
        explaination = te.explain_prediction(target_names=list(explainer_generator.idx2tag.values()),top_targets=3)
        exp = eli5.formatters.format_as_dict(explaination)
        features = exp['targets']
        print("----------------------------------------------")
        print("Model Fitting Completed")
  
        for item in features:
            for ite in item['feature_weights']['pos']:
                word = ite['feature'].split()
                del(word[0])
                word = ''.join(word)
                if item['target'] == '|B-DISEASE\n':
                    word_importance["B"].append(word)           
                elif item['target'] == '|I-DISEASE\n':
                    word_importance["I"].append(word)           
                elif item['target'] == '|O\n':
                    word_importance["O"].append(word)           
    except Exception as e:
        print(e)

        # Words with negative weights
        # for ite in item['feature_weights']['neg']:
        #     word = ite['feature'].split()
        #     del(word[0])
        #     word = ''.join(word)
        #     if item['target'] == '|B-DISEASE\n':
        #         word_importance["B"].append(word)           
        #     elif item['target'] == '|I-DISEASE\n':
        #         word_importance["I"].append(word)           
        #     elif item['target'] == '|O\n':
        #         word_importance["O"].append(word)           
        
    return word_importance