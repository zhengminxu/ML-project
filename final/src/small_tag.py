from sklearn.feature_extraction import text
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from gensim.models import Word2Vec
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
#import tensorflow as tf
from collections import Counter
import numpy as np
import pandas as pd
import string
import logging
import sys #argv[1] is the path containing all the data
import re
import os.path
import random

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
embed_SIZE = 100

class process_features:

	def __init__(self, args):
		#paths
		assert len(args) == 2, "Unmatched size of arguments, should be \'data path\' and \'word_model name\'"
		#data
		self.data = self.load_data(args[0])
		#model
		self.word_model = []
		self.sent_embed(args[1])
		#tags
		self.tagset = []
		self.val_tagset = []
		self.get_tags()
		self.maxtag = 1000

	def load_data(self, data_path):
		print "Loading data..."
		topics = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
		data = []
		for x in topics:
			data.append( pd.read_csv( data_path + "/" + x + ".csv") )
		print "Done!"
		return data

	def sent_embed(self, word_model_name):
		path = "../model/" + word_model_name
		if os.path.isfile(path):
			print "Model already exists, Loading in..."
			self.word_model = Word2Vec.load(path)
		else:
			data = self.get_content()
			#tags, _ = self.get_tags()
			#introduce label_data
			#data += tags
			print "Building Word2Vec model"
			self.word_model = Word2Vec(size=embed_SIZE, alpha=0.002, min_alpha=0.0005, window=8, \
				min_count=8, sample=1e-4, workers=12, sg=1, hs=1, iter=5)
			self.word_model.build_vocab(data)
			self.word_model.train(data)
			print "Saving wordvec model to directory model..."
			self.word_model.save(path)
		print "Done!"
		print "Test : most similar words with 'biology' %s" %self.word_model.most_similar(positive=['biology'])

	def get_train(self):
		assert self.word_model, "Haven't initialized word model ! "
		(x_train, x_val) = self.get_content(False)
		X_train, Y_train, X_val, Y_val = [], [], [], []
		for x in x_train:
			if x in self.word_model:
				if x in self.tagset:
					Y_train += [self.tagset[x] / self.maxtag if self.tagset[x] <= self.maxtag else self.maxtag]
				else:
					Y_train += [0]
				X_train += [self.word_model[x]]
		for x in x_val:
			if x in self.word_model:
				if x in self.val_tagset:
					if x == 'visa':
						print "QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"
					Y_val += [self.val_tagset[x] / self.maxtag if self.val_tagset[x] <= self.maxtag else self.maxtag]
				else:
					Y_val += [0]
				X_val += [self.word_model[x]]
		#do shuffle
		#combined = list(zip(x_train, y_train))
		#random.shuffle(combined)
		#x_train[:], y_train[:] = zip(*combined)
		x_train = np.asarray(X_train)
		y_train = np.asarray(Y_train)
		print y_train
		x_val = np.asarray(X_val)
		y_val = np.asarray(Y_val)
		return x_train, x_val, y_train, y_val
	# dim : (topics, sentences, words) or (topics, contents, words)

	########### Helper Functions ###########

	def clean_html(self, text):
		return [re.sub('<.*?>', '', x) for x in text]

	def get_token(self, sent):
		sent = sent.decode('utf-8')
		lmtzr = WordNetLemmatizer()
		#doing spliting and keeping '-' i.e. ['ribosome', 'binding', '-', 'sites', 'translation', 'synthetic', '-', 'biology']
		#reg = re.compile(r'[^a-zA-Z0-9-]?')
		#punc_set = string.punctuation.replace("-", "")
		#data = [x.strip() for x in re.split(reg, sent) if (x.strip() and (x.strip() not in punc_set))]
		data = [x for x in re.split(r'(\W)', sent) if x.strip()]
		cond = lambda x: len(x) > 1 #or x == '-'
		data = [x.lower() for x in data]
		data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
		data = [lmtzr.lemmatize(x) for x in data if cond(x)]
		data = [lmtzr.lemmatize(x, pos='v') for x in data if cond(x)] #not sure
		data = [x for x in data if cond(x)]
		return data

	def get_tags(self):
		datas = self.data
		print "Getting tags..."
		tags, val_tags = [], []
		for idx, x in enumerate(datas):
			data = x.tags.values.tolist()
			if idx != (len(datas) - 1):
				for x in data:
					tags += self.get_token(x)
			else:
				for x in data:
					val_tags += self.get_token(x)
		self.tagset = Counter(tags)
		self.val_tagset = Counter(val_tags)

	def clean_pos(self, data):
		p = pos_tag(data)
		cond = lambda x: x[1] == 'JJ' or x[1] == 'NN' #or x[1] == ':'
		data = [data[i] for i, x in enumerate(p) if cond(x)]
		return data

	def get_content(self, word2vec_or_not=True):
		datas = self.data
		print "Parsing data..."
		all_datas = []
		for idx, x in enumerate(datas):
			data = x.content.values.tolist()
			data = self.clean_html(data)
			if word2vec_or_not:
				data = [x.split('\n') for x in data]
				data = [self.get_token(w) for sent in data for w in sent if self.get_token(w)]
				all_datas += data
			else:
				temp = []
				data = [re.sub(r'\n', ' ', x) for x in data]
				for d in data:
					if self.get_token(d):
						temp += self.get_token(d)
				print idx, '/', len(datas) - 1
				if idx != (len(datas) - 1):
					all_datas += temp
				else:
					val_data = temp
		print "Done!"
		if word2vec_or_not:
			return all_datas
		else:
			th = 50
			a = Counter(all_datas)
			b = Counter(val_data)
			all_datas = list(set(all_datas))
			val_data = list(set(val_data))
			temp, tempv = [], []
			for x in all_datas:
				if a[x] > th:
					temp += [x] * int(a[x] / th)
			for x in val_data:
				if b[x] > th:
					tempv += [x] * int(b[x] / th)
			all_datas = self.clean_pos(temp)
			val_data = self.clean_pos(tempv)
			print val_data
			return (all_datas, val_data)


feat = process_features(sys.argv[1:])
x_train, x_val, y_train, y_val = feat.get_train()

model = Sequential()
model.add(Dense(64, input_shape=(embed_SIZE,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',\
	optimizer=Adam(lr=0.0001),\
	metrics=['accuracy'])

model.fit(x_train, y_train,\
	batch_size=128,\
	nb_epoch=100,\
	validation_data=(x_val, y_val))

p = model.predict(x_val, batch_size=32)
p = [feat.word_model.most_similar(positive=[x_val[i]], topn=1)[0][0] for i, x in enumerate(p) if x > 0.8]
print p

def myf1(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=1)

#TODO
#POS tagging keeps only adj, noun.
#then change the tagets to one hot encoding of output words(think that with the split of '-', the OOV will be rare)
#apply another NN for classfication of where '-' should involve in
#This will not be end-to-end.
#Q : better approach?

#TODO
#first filter out words that maybe tags(regression one by one).
#use the reduced set of tags to train...
