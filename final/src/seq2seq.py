from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import text
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
import numpy as np
import pandas as pd
import string
import logging
import sys #argv[1] is the path containing all the data
import re
import os.path
import cPickle
import random

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
embed_SIZE = 100
#end of taggings
EOF = np.full(embed_SIZE, 1.0)
#padding pattern
PAD = np.full(embed_SIZE, 0.0)
#pad to length
weights = 0
MAXLEN_X = 200
MAXLEN_Y = 10 # 9tags + EOF

def load_data():
	print "Loading data..."
	topics = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
	data = []
	for x in topics:
		data.append( pd.read_csv( sys.argv[1] + "/" + x + ".csv") )
	print "Done!"
	return data

def clean_html(text):
	return [re.sub('<.*?>', '', x) for x in text]

def get_token(sent, tags=False):
	sent = sent.decode('utf-8')
	lmtzr = WordNetLemmatizer()
	if not tags:
		reg = re.compile(r'[^a-zA-Z0-9-]?')
		punc_set = string.punctuation.replace("-", "")
		data = [x.strip() for x in re.split(reg, sent) if (x.strip() and (x.strip() not in punc_set))]
	else: #doing spliting and keeping '-' i.e. ['ribosome', 'binding', '-', 'sites', 'translation', 'synthetic', '-', 'biology']
		data = [x for x in re.split(r'(\W)', sent) if x.strip()]
	data = [x.lower() for x in data]
	data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
	data = [lmtzr.lemmatize(x) for x in data if (len(x) > 1 or x == '-')]
	data = [lmtzr.lemmatize(x, pos='v') for x in data if (len(x) > 1 or x == '-')] #not sure
	data = [x for x in data if (len(x) > 1 or x =='-')]
	return data

# dim : (topics, sentences, words) or (topics, contents, words)
def get_content(word2vec_or_not=True):
	datas = load_data()
	print "Parsing data..."
	all_datas = []
	for idx, x in enumerate(datas):
		data = x.content.values.tolist()
		data = clean_html(data)
		if word2vec_or_not:
			data = [x.split('\n') for x in data]
			data = [get_token(w) for sent in data for w in sent if get_token(w)]
			all_datas += data
		else:
			temp = []
			data = [re.sub(r'\n', ' ', x) for x in data]
			#assign 'meat' for poor ones that have only stop words
			#(just in case, a little chance that this will happen)
			y = [get_token(d) if get_token(d) else ['meat'] for d in data]
			print idx, '/', len(datas) - 1
			if idx is not len(datas) - 1:
				all_datas += y
			else:
				val_data = y

	print "Done!"
	if word2vec_or_not:
		return all_datas
	else:
		return (all_datas, val_data)

def get_tags():
	global MAXLEN_Y
	datas = load_data()
	print "Getting tags..."
	all_tags = []
	for idx, x in enumerate(datas):
		data = x.tags.values.tolist()
		temp = []
		for x in data:
			temp.append( get_token(x, True) )
		data = temp
	
		if idx is not len(datas) - 1:
			all_tags += data
		else:
			val_tags = data
	print "Done!"
	MAXLEN_Y = len(max(all_tags + val_tags, key=len))
	print "Longest tags : ", MAXLEN_Y
	return (all_tags, val_tags)

def sent_embed():
	if os.path.isfile("../model/" + sys.argv[2]):
		print "Model already exists, Loading in..."
		model = Word2Vec.load("../model/" + sys.argv[2])
	else:
		data = get_content()
		tags, _ = get_tags()
		#introduce label_data
		data += tags
		print "Building Word2Vec model"
		model = Word2Vec(size=embed_SIZE, alpha=0.002, min_alpha=0.0005, window=8, \
			min_count=8, sample=1e-4, workers=12, sg=1, hs=1, iter=5)
		model.build_vocab(data)
		model.train(data)
		print "Saving wordvec model to directory model..."
		model.save("../model/" + sys.argv[2])
	print "Done!"
	print "Test : most similar words with 'biology' %s" %model.most_similar(positive=['biology'])
	return model

def do_embed(x_, y_, model):
	print "Encoding..."
	z = []
	for idx, x in enumerate(x_):
		#print float(idx)/length * 100.0, '%'
		buf = [model[w] for w in x if w in model] #one content
		z.append(buf)
	x_ = z
#do y_
#zero padding to max_len
	maxlen = len(max(y_, key=len))
	#MAXLEN_Y = maxlen
	y = []
	for tags in y_:
		temp = []
		for tag in tags:
			if tag in model:
				temp += [model[tag]]
		y.append(temp)
	y_ = y
	print len(x_), len(y_)
	
	print "Done"
	return (x_, y_)

def get_train():
	model = sent_embed()
	(x_train, x_val) = get_content(False)
	(y_train, y_val) = get_tags()
	print len(y_val), len(x_val)
	(x_train, y_train) = do_embed(x_train, y_train, model)
	(x_val, y_val) = do_embed(x_val, y_val, model)
	#do shuffle
	combined = list(zip(x_train, y_train))
	random.shuffle(combined)
	x_train[:], y_train[:] = zip(*combined)
	return x_train, x_val, y_train, y_val

def embed_content(in_shape, outlength):
	model = Sequential()
	model.add(LSTM(64, input_shape=in_shape))
	model.add(Dense(32, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	#decoder
	model.add(RepeatVector(outlength))
	model.add(LSTM(64, return_sequences=True))
	model.add(TimeDistributed(Dense(embed_SIZE, activation='softmax')))
	return model

def data_gen_train(x_train, y_train, n):
	step = len(x_train) / n
	print step
	w = [[1.0] * (len(c) + 1) + [0.0] * (MAXLEN_Y - 1 - len(c)) if MAXLEN_Y - 1 > len(c) else [1.0] * MAXLEN_Y for c in y_train]
	a = [c + [PAD] * (MAXLEN_X - len(c)) if MAXLEN_X > len(c) else c[:MAXLEN_X] for c in x_train]
	b = [c + [EOF] + [PAD] * (MAXLEN_Y - 1 - len(c)) if MAXLEN_Y - 1 > len(c) else c[:MAXLEN_Y - 1] + [EOF] for c in y_train]
	while 1:
		print ("generating training data")
		for x in range(0, len(x_train), step):
			if x + step > len(x_train):
				x = len(x_train) - step
			X = np.asarray(a[x:x + step])
			Y = np.asarray(b[x:x + step])
			W = np.asarray(w[x:x + step])
			yield (X, Y, W)

def data_gen_val(x_val, y_val, n):
	step = len(x_val) / n
	w = [[1.0] * (len(c) + 1) + [0.0] * (MAXLEN_Y - 1 - len(c)) if MAXLEN_Y - 1 > len(c) else [1.0] * MAXLEN_Y for c in y_val]
	a = [c + [PAD] * (MAXLEN_X - len(c)) if MAXLEN_X > len(c) else c[:MAXLEN_X] for c in x_val]
	b = [c + [EOF] + [PAD] * (MAXLEN_Y - 1 - len(c)) if MAXLEN_Y - 1 > len(c) else c[:MAXLEN_Y - 1] + [EOF] for c in y_val]
	while 1:
		print ("generating training validation data")
		for x in range(0, len(x_val), step):
			if x + step > len(x_val):
				x = len(x_val) - step
			X = np.asarray(a[x:x + step])
			Y = np.asarray(b[x:x + step])
			W = np.asarray(w[x:x + step])
			yield (X, Y, W)

def myf1(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=1)

def train_model():
	x_train, x_val, y_train, y_val = get_train()
	X_SHAPE = (MAXLEN_X, embed_SIZE)
	model = embed_content(X_SHAPE, MAXLEN_Y)
	model.compile(loss='categorical_crossentropy',\
		optimizer=Adam(lr=0.0008),\
		sample_weight_mode="temporal",\
		metrics=['accuracy', 'cosine_proximity'])
	model.fit_generator(data_gen_train(x_train, y_train, 677),\
		samples_per_epoch=20000,\
		nb_epoch=40,\
		validation_data=data_gen_val(x_val, y_val, 677),\
		nb_val_samples=20000)

def get_result(model, y_pred):
	model = Word2Vec.load(wmodel_path + sys.argv[1])
	model.most_similar(positive=[y_pred])

train_model()
#TODO
#POS tagging keeps only adj, noun.
#then change the tagets to one hot encoding of output words(think that with the split of '-', the OOV will be rare)
#apply another NN for classfication of where '-' should involve in
#This will not be end-to-end.
#Q : better approach?
