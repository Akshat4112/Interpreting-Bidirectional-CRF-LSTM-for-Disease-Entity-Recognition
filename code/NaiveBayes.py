''' 
Project: Explaining LSTM-CRF models based NER Systems
Version: 0.1
Author: Akshat Gupta
'''


# Importing important libraries
from collections import Counter  #for counting items in the list and store them as a dict
import math  #for performing mathematical operations
import time  #for computing time taken by our code snippets

# Naive Bates Class has from scratch implementation of Naive Bayes Algorithm

class NaiveBayes(object):
    def __init__(self):
        return None

    # MultinomialNBTrain() which is responsible for training on the data.    
    def MultinomialNBTrain(self, X, y):
        print("Named Entity Recognition using Naive Bayes Classifier.")
        print("Training Started...")

        #Putting some Animation! 
        animation = [
        "[        ]",
        "[=       ]",
        "[===     ]",
        "[====    ]",
        "[=====   ]",
        "[======  ]",
        "[======= ]",
        "[========]",
        "[ =======]",
        "[  ======]",
        "[   =====]",
        "[    ====]",
        "[     ===]",
        "[      ==]",
        "[       =]",
        "[        ]",
        "[        ]"
        ]

        notcomplete = True

        i = 0

        while notcomplete:
            print(animation[i % len(animation)], end='\r')
            time.sleep(.1)
            i += 1
            if i == 1*1:
                break


        self.X  = X
        self.y = y

        # Computing class counts
        count_classes = Counter(y)
        count_B = count_classes['|B-DISEASE\n']
        count_I = count_classes['|I-DISEASE\n']
        count_O = count_classes['|O\n']
        doc_count = count_B + count_I + count_O
        
        self.features = {}
        self.features['B'] = {}
        self.features['I'] = {}
        self.features['O'] = {}
        
        # Computing prior probabilities
        self.priori_B = math.log(count_B/ doc_count)
        self.priori_I = math.log(count_I/ doc_count)
        self.priori_O = math.log(count_O/ doc_count)

        B = []
        I = []
        O = []

        for item, label in zip(X,y):
            if label == '|B-DISEASE\n':
                B.append(item)
            elif label == '|I-DISEASE\n':
                I.append(item)
            else:
                O.append(item)

        B_dict = Counter(B)
        I_dict = Counter(I)
        O_dict = Counter(O)

        for word, count in B_dict.items():
            self.features['B'][word] = math.log((int(count) + 1) /(count_B + doc_count))
        for word, count in I_dict.items():
            self.features['I'][word] = math.log((int(count) + 1) /(count_I + doc_count))
        for word, count in O_dict.items():
            self.features['O'][word] = math.log((int(count) + 1) /(count_O + doc_count))
        print("Training Completed...")

    # MultinomialNBTest() which is responsible for the prediction of entities using Naives Bayes trained model.
    def MultinomialNBTest(self, X):
        self.X = X
        p_B, p_I, p_O = 0 ,0, 0

        # Add word based probability to get the compelte prior for the class
        p_B += self.priori_B
        p_I += self.priori_I
        p_O += self.priori_O
        
        # Adding the likelihood to the prior based on class
        for item in self.X:
            if item in self.features['B']:
                p_B += self.features['B'][item]

        for item in self.X:
            if item in self.features['I']:
                p_I += self.features['I'][item]

        for item in self.X:
            if item in self.features['O']:
                p_O += self.features['O'][item]

        # We take the label with highest probability as a prediction
        results = [p_B, p_I, p_O]
        max_value = max(results)
        max_index = results.index(max_value)

        # Returning the class label
        if max_index == 0:
            return "|B-DISEASE\n"
        elif max_index == 1:
            return "|I-DISEASE\n"
        else:
            return "|O\n"


    def MultinomialNBTrainGene(self, X, y):
        print("Named Entity Recognition using Naive Bayes Classifier.")
        print("Training Started...")

        #Putting some Animation! 
        animation = [
        "[        ]",
        "[=       ]",
        "[===     ]",
        "[====    ]",
        "[=====   ]",
        "[======  ]",
        "[======= ]",
        "[========]",
        "[ =======]",
        "[  ======]",
        "[   =====]",
        "[    ====]",
        "[     ===]",
        "[      ==]",
        "[       =]",
        "[        ]",
        "[        ]"
        ]

        notcomplete = True

        i = 0

        while notcomplete:
            print(animation[i % len(animation)], end='\r')
            time.sleep(.1)
            i += 1
            if i == 1*1:
                break


        self.X  = X
        self.y = y

        # Computing class counts
        count_classes = Counter(y)
        count_B = count_classes['|B-PROTEIN\n']
        count_I = count_classes['|I-PROTEIN\n']
        count_O = count_classes['|O\n']
        doc_count = count_B + count_I + count_O
        
        self.features = {}
        self.features['B'] = {}
        self.features['I'] = {}
        self.features['O'] = {}
        
        # Computing prior probabilities
        self.priori_B = math.log(count_B/ doc_count)
        self.priori_I = math.log(count_I/ doc_count)
        self.priori_O = math.log(count_O/ doc_count)

        B = []
        I = []
        O = []

        for item, label in zip(X,y):
            if label == '|B-PROTEIN\n':
                B.append(item)
            elif label == '|I-PROTEIN\n':
                I.append(item)
            else:
                O.append(item)

        B_dict = Counter(B)
        I_dict = Counter(I)
        O_dict = Counter(O)

        for word, count in B_dict.items():
            self.features['B'][word] = math.log((int(count) + 1) /(count_B + doc_count))
        for word, count in I_dict.items():
            self.features['I'][word] = math.log((int(count) + 1) /(count_I + doc_count))
        for word, count in O_dict.items():
            self.features['O'][word] = math.log((int(count) + 1) /(count_O + doc_count))
        print("Training Completed...")

    # MultinomialNBTest() which is responsible for the prediction of entities using Naives Bayes trained model.
    def MultinomialNBTestGene(self, X):
        self.X = X
        p_B, p_I, p_O = 0 ,0, 0

        # Add word based probability to get the compelte prior for the class
        p_B += self.priori_B
        p_I += self.priori_I
        p_O += self.priori_O
        
        # Adding the likelihood to the prior based on class
        for item in self.X:
            if item in self.features['B']:
                p_B += self.features['B'][item]

        for item in self.X:
            if item in self.features['I']:
                p_I += self.features['I'][item]

        for item in self.X:
            if item in self.features['O']:
                p_O += self.features['O'][item]

        # We take the label with highest probability as a prediction
        results = [p_B, p_I, p_O]
        max_value = max(results)
        max_index = results.index(max_value)

        # Returning the class label
        if max_index == 0:
            return "|B-PROTEIN\n"
        elif max_index == 1:
            return "|I-PROTEIN\n"
        else:
            return "|O\n"            