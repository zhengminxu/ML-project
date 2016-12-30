from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from keras.models import Sequential, Model
from keras.models import load_model
from collections import Counter
from scipy import sparse
from util import get_token, clean_html 
import numpy as np
import pandas as pd
import string
import sys #argv[1] is the path containing all the data
import re
import os.path
import gc

embed_SIZE = 700

def push_dict(dic, w):
	if w in dic:
		dic[w] += 1
	else:
		dic[w] = 1

def get_bigram(x_test_c, raw_tags):
	counter = {}
	freq_th = 100
	for doc in x_test_c:
		for i, w in enumerate(doc):
			if w in raw_tags:
				if i != 0:
					front = (doc[i-1], w)
					push_dict(counter, front)
				if i != len(doc) - 1:
					back = (w, doc[i+1])
					push_dict(counter, back)
	counter = Counter(counter)
	print counter
	return [k for k, v in counter.iteritems() if v >= freq_th]

def get_trigram(x_test_c, raw_tags):
	counter = {}
	freq_th = 100
	for doc in x_test_c:
		for i, w in enumerate(doc):
			if w in raw_tags:
				if i > 1:
					front = (doc[i-2], doc[i-1], w)
					push_dict(counter, front)
				if i < len(doc) - 2:
					back = (w, doc[i+1], doc[i+2])
					push_dict(counter, back)
				if i != 0 and i != len(doc) - 1:
					middle = (doc[i-1], w, doc[i+1])
					push_dict(counter, middle)
	counter = Counter(counter)
	print counter
	return [k for k, v in counter.iteritems() if v >= freq_th]

def get_test(data_path):
	print "Getting physics..."
	data = pd.read_csv( data_path + "/test.csv")
	data = data.content.values.tolist()
	data = clean_html(data)
	data = [re.sub(r'\n', ' ', x) for x in data]
	temp = []
	#for x in data:
	#	temp += x.split()
	#c = Counter(temp)
	#c = {k:v for k, v in c.iteritems() if k not in text.ENGLISH_STOP_WORDS}
	#c = Counter(c)
	gc.collect()
	dic = {}
	temp = []
	for d in data:
		a, dic = get_token(d, dic)
		if a:
			temp += [a]
	dic = { key:max(set(value), key=value.count) for key, value in dic.iteritems() }
	mul = 10
	x_test = [x for sublist in temp for x in sublist]
	c = Counter(x_test)
	most_set = [x[0] for x in c.most_common(20)]
	x_test = list(set(x_test))
	x_test = [x for x in x_test if c[x] >= 200]
	txt = open('../feat/test_feats', 'w')
	print >> txt, (temp, x_test, dic, most_set)
	return temp, x_test, dic, most_set 

def process_test(x_test_c=None, x_test=None, feat_path=None):
	if feat_path is not None:
		return np.load(feat_path)
	else:
		assert x_test_c is not None, "Please specify x_test_c."
		assert x_test is not None, "Please specify x_test."
	ll = lambda x: float(len(x))
	lt = map(ll, x_test_c)
	mul = 10.0
	qq = 0
	for w in x_test:
		if not qq:
			z = np.array([doc.count(w) * mul / lt[i] for i, doc in enumerate(x_test_c)])
			qq = 1 
		else:
			z = np.vstack((z, np.array([doc.count(w) * mul / lt[i] for i, doc in enumerate(x_test_c)])))
	x_test = z 
	gc.collect()
	print "Doing LSA"
	print "SVD...."
	u, s, v = sparse.linalg.svds(x_test, embed_SIZE)
	n = Normalizer(copy=False)
	x_test = n.fit_transform(u * s.transpose())
	np.save('../feat/svd.npy', x_test)
	return x_test

def clean_dup_bigram(bigram_tags, trigram_tags):
	any_same = lambda x, y: (x[0] == y[0] and x[1] == y[1]) \
								or (x[0] == y[1] and x[1] == y[2])
	for x in trigram_tags:
		for y in bigram_tags:
			if any_same(y, x):
				del y
	return bigram_tags

def get_physics_tags(data_path, model_path):
	print "loading Model..."
	model = load_model('../model/best.h5')
	print "Getting test..."
	x_test_c, tags, dic, most_set = get_test(data_path)
	x_test = process_test(x_test_c, tags)
	tags = np.array(tags)
	n_large = 300
	p = model.predict(x_test, batch_size=64)
	p = np.array([x[0] for x in p])
	out = np.argpartition(p, -1 * n_large)[-1 * n_large:]
	out = out[np.argsort(p[out])[::-1]]
	tags = tags[out].tolist()
	tags += most_set
	trigram_tags = get_trigram(x_test_c, tags)
	any_same = lambda a, b, c: len(set([a, b, c])) == 3
	trigram_tags = [(dic[a], dic[b], dic[c]) for a, b, c in trigram_tags if any_same(a, b, c)]
	trigram_tags = list(set(trigram_tags))
	bigram_tags = get_bigram(x_test_c, tags)
	bigram_tags = [(dic[a], dic[b]) for a, b in bigram_tags if a != b]
	bigram_tags = list(set(bigram_tags))
#	bigram_tags = clean_dup_bigram(bigram_tags, trigram_tags)
	tags = [dic[t] for t in tags]
	tags = list(set(tags))
	tags = [x for x in tags if len(x) > 4]
	print tags, bigram_tags
	return tags, bigram_tags, trigram_tags

if __name__ == '__main__':
	assert len(sys.argv) == 3, "Unmatched size of sys.argv !"
	tags, bi_tags, tri_tags = get_physics_tags(sys.argv[1], sys.argv[2])
	txt = open('../feat/tags', 'w')
	print >> txt, (tags, bi_tags, tri_tags)

#TODO
#POS tagging keeps only adj, noun.
#then change the tagets to one hot encoding of output words(think that with the split of '-', the OOV will be rare)
#apply another NN for classfication of where '-' should involve in
#This will not be end-to-end.
#Q : better approach?

#TODO
#first filter out words that maybe tags(regression one by one).
#use the reduced set of tags to train...
