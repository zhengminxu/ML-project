from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from gensim.models import Word2Vec
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
embed_SIZE = 500

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
	else:
		data = sent.split()
	data = [x.lower() for x in data]
	data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
	data = [lmtzr.lemmatize(x) for x in data if len(x) > 1]
	data = [lmtzr.lemmatize(x, pos='v') for x in data if len(x) > 1] #not sure
	data = [x for x in data if len(x) > 1]
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
	return (all_tags, val_tags)

def sent_embed():
	if os.path.isfile("../model/" + sys.argv[2]):
		print "Model already exists, Loading in..."
		model = Word2Vec.load("../model/" + sys.argv[2])
	else:
		data = get_content()
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
	meat_count = 0
	length = float(len(zip(x_, y_)))
	for idx, (x, y) in enumerate(zip(x_, y_)):
		#print float(idx)/length * 100.0, '%'
		buf = [model[w] for w in x if w in model] #one content
		if buf:
			z += ([buf] * len(y))
		else: #for those who only contains words less than min count (rare)
			z += ([[model['meat']]] * len(y))
			meat_count += 1
	x_ = z
	
	y_ = [model[tag] if tag in model else model['meat'] \
		for tags in y_ for tag in tags]
	print len(x_), len(y_)
	
	print "Total zero contents : ", meat_count
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
	return (x_train, x_val, y_train, y_val)

train = get_train()
print "Saving training features to directory feat..."
f = open("../feat/train", 'wb')
cPickle.dump(train, f, protocol=cPickle.HIGHEST_PROTOCOL)
