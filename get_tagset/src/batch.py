from sklearn.feature_extraction import text
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from keras.models import Sequential, Model
from keras.models import load_model
from collections import Counter
from scipy import sparse
import numpy as np
import pandas as pd
import string
import sys #argv[1] is the path containing all the data
import re
import os.path
import gc

embed_SIZE = 700

class process_features:

	def __init__(self, args):
		#paths
		assert len(args) == 1, "Unmatched size of arguments, should be \'data path\' "
		#data
		self.data_path = args[0]

	def get_test(self):
		print "Getting physics..."
		data = pd.read_csv( self.data_path + "/test.csv")
		data = data.content.values.tolist()
		data = self.clean_html(data)
		temp = []
		data = [re.sub(r'\n', ' ', x) for x in data]
		for d in data:
			a = self.get_token(d)
			if a:
				temp += [a]
		mul = 10
		x_test = [x for sublist in temp for x in sublist]
		c = Counter(x_test)
		print c
		x_test = list(set(x_test))
		x_test = [x for x in x_test if c[x] >= 150]
		ll = lambda x: float(len(x))
		lt = map(ll, temp)
		print len(x_test)
		n = Normalizer(copy=False)
		batch_size = 700
		gc.collect()
		for a in range(0, len(x_test), batch_size):
			if a + batch_size > len(x_test):
				x = x_test[a:len(x_test)]
			else:
				x = x_test[a:a + batch_size]
			x = [[doc.count(w) * mul / lt[i] for i, doc in enumerate(temp)] for w in x]
			x = np.array(x)
			#x_train = np.concatenate((x_train, np.zeros((x_train.shape[0], self.doclen - x_train.shape[1]))), axis=1)
			print "Doing LSA"
			print "SVD...."
			print x.shape
			u, s, v = sparse.linalg.svds(x, embed_SIZE)
			x = n.fit_transform(u * s.transpose())
			txt = open('../feat/test', 'a')
			print >> txt, x
			yield x
	
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
		reg = re.compile(r'[^a-zA-Z]?')
		data = [x for x in re.split(reg, sent) if x.strip()]
		cond = lambda x: len(x) > 3 and len(x) <= 20 #or x == '-'
		data = [x.lower() for x in data if not x.isdigit()]
		data = self.clean_pos(data)
		data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
		data = [lmtzr.lemmatize(x) for x in data if cond(x)]
#		data = [lmtzr.lemmatize(x, pos='v') for x in data if cond(x)] #not sure
		data = [x for x in data if cond(x)]
		return data

	def clean_pos(self, data):
		p = pos_tag(data)
		cond = lambda x: x[1] == 'JJ' or x[1] == 'NN' #or x[1] == ':'
		data = [data[i] for i, x in enumerate(p) if cond(x)]
		return data

print "asd"
feat = process_features(sys.argv[1:])
print "loading Model..."
model = load_model('../model/best.h5')
print "Getting test..."
n_large = 50
p = []
bs = 50
while 1:
	try:
		b = bs
		x_test = next(feat.get_test())
		if x_test.shape[0] < bs:
			b = x_test.shape[0]
		pred = model.predict(x_test, batch_size=b)
		p += [x[0] for x in pred]
	except StopIteration:
		break
p = np.array(p)
out = np.argpartition(p, -1 * n_large)[-1 * n_large:]
out = out[np.argsort(p[out])[::-1]]
t = tags[out]
print t


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
