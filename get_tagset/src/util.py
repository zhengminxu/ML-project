from sklearn.feature_extraction import text
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
import random
import string
import re

def prob_bool(p):
   assert p >= 0 and p <= 1, "Probability required to be between 0 and 1 !"
   return random.uniform(0, 1) < p

def clean_html(text):
   return [re.sub('<.*?>', '', x) for x in text]

def clean_pos(data):
   p = pos_tag(data)
   cond = lambda x: x[1] == 'JJ' or x[1] == 'NN' or x[1] == 'NNS'
   data = [data[i] for i, x in enumerate(p) if cond(x)]
   return data

def get_token(sent, dic):
   sent = sent.decode('utf-8')
   lmtzr = WordNetLemmatizer()
   reg = re.compile(r'[^a-zA-Z\\]?')
   cond = lambda x: x.strip() and '\\' not in x
   data = [x for x in re.split(reg, sent) if cond(x)]
   cond = lambda x: len(x) > 3 and len(x) <= 20 #or x == '-'
   data = [x.lower() for x in data if not x.isdigit()]
   data = clean_pos(data)
   data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
   d = []
   for x in data:
      if cond(x):
         lemmed = lmtzr.lemmatize(x)
         d += [lemmed]
         if lemmed in dic:
            if x[-2:] != 'al':
               if dic[lemmed][-1][-2:] == 'al':
                  del dic[lemmed][-1]
               dic[lemmed] += [x]
         else:
            dic[lemmed] = [x]
   data = d
   data = [x for x in data if cond(x)]
   return data, dic
