import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import operator
from nltk import FreqDist
import sys
import csv

## Read csv
## target: the corpus
## out_old: original output
## out_new: new output
## nb_cluster: number of clusters
target = sys.argv[1]
corpus_pd = pd.read_csv(target)
if (len(sys.argv) < 3):
	out_old = 'outputs/' + target.split('.')[0] + '_est.csv'
	out_new = 'outputs/' + target.split('.')[0] + '_est2.csv'
else:
	out_old = 'outputs/' + sys.argv[2]
	out_new = 'outputs/' + sys.argv[2].split('.')[0] + '_tmp.csv'

if (len(sys.argv) == 4):
	nb_cluster = int(sys.argv[3])
else:
	nb_cluster = 500

## Make 'title' + ' ' + 'contents' string list
def remove_html(context):
	cleaner = re.compile('<.*?>')
	clean_text = re.sub(cleaner,'',context)
	return clean_text

ids = corpus_pd.id.values.tolist()
titles = corpus_pd.title.values.tolist()
contents = corpus_pd.content.values.tolist()
tags_ans = corpus_pd.tags.values.tolist()
contents = [remove_html(c) for c in contents]
contents = [re.sub(r'\n', ' ', c) for c in contents]
corpus = [titles[i]+' '+contents[i] for i in xrange(len(ids))]

## Read previous estimation
corpus_est = pd.read_csv(out_old)
tags_est = corpus_est.tags.values.tolist()

## Tokenize string
split_regex = r'[^a-zA-Z_]'
my_stopwords = set(stopwords.words('english'))

def tokenize(string):
	return [token for token in re.split(split_regex, string.lower()) if token and (not token in my_stopwords)]

corpus_tok = [tokenize(s) for s in corpus]

## Regularize the words
### Lemmatize?
lmtzr = WordNetLemmatizer()
corpus_lemmed = [[lmtzr.lemmatize(s) for s in tokens] for tokens in corpus_tok]
### Stem?
stemmer = PorterStemmer()
corpus_stemmed = [[stemmer.stem(s) for s in tokens] for tokens in corpus_tok]

## TF-IDF
vectorizer = TfidfVectorizer()
corpus_raw = [" ".join(s) for s in corpus_stemmed]
corpus_tfidf = vectorizer.fit_transform(corpus_raw)

## LSA
u, s, v = sparse.linalg.svds(corpus_tfidf, 200)
n = Normalizer(copy=False)
feat = n.fit_transform(u * s.transpose())

## K-Means
# nb_cluster = 1000
# km = KMeans(n_clusters=nb_cluster, n_init=10, init='k-means++', max_iter=200, verbose=2)
# km.fit(feat)


def frequent(context):
	freq = FreqDist(context)
	return freq

### Print the most 3 frequent words, tags, and estimations in each cluster
### For debug only
# def print_tags_in_cluster(cl_start):
# 	for cl in xrange(cl_start,cl_start+10,1):
# 		temp_tags = {}
# 		temp_est = {}
# 		n_items = 0
# 		### Also compute all words in all documents of this cluster
# 		all_words_ = []
# 		for idx in xrange(len(km.labels_)):
# 			if (km.labels_[idx] == cl):
# 				n_items = n_items + 1
# 				for tag in re.split(' ', tags[idx]):
# 					temp_tags[tag] = temp_tags.get(tag, 0) + 1
# 				# print '   ', tags[idx]
# 				if not type(tags_est[idx]) is float:
# 					for tag in re.split(' ', tags_est[idx]):
# 						temp_est[tag] = temp_est.get(tag, 0) + 1
# 				# print '      ', tags_est[idx]
# 				all_words_.append([t for t in corpus_lemmed[idx]])
# 		all_words = frequent([w for w_list in all_words_ for w in w_list if len(w) > 2])
# 		print "cluster", cl, 'has', n_items, 'items'
# 		print '   Frq:', all_words.most_common(3)
# 		n_items = float(n_items)
# 		temp_tags = sorted(temp_tags.items(), key=operator.itemgetter(1), reverse = True)
# 		print '   Ans:', [(a,b/n_items) for a,b in temp_tags[0:3]]
# 		temp_est = sorted(temp_est.items(), key=operator.itemgetter(1), reverse = True)
# 		print '   Est:', [(a,b/n_items) for a,b in temp_est[0:3]]

# print_tags_in_cluster(0)

## Run 5 times to find the most probable tag from its neighbors
corpus_common_tags = {}
for idx in xrange(len(ids)):
	corpus_common_tags[idx] = []

for iter in xrange(5):
	km = KMeans(n_clusters=nb_cluster, n_init=10, init='k-means++', max_iter=200, verbose=0)
	km.fit(feat)
	for cl in xrange(nb_cluster):
		temp_est = []
		# n_items = 0
		for idx in xrange(len(km.labels_)):
			if (km.labels_[idx] == cl):
				# n_items = n_items + 1
				if not type(tags_est[idx]) is float:
					for tag in re.split(' ', tags_est[idx]):
						temp_est.append(tag)
		for idx in xrange(len(km.labels_)):
			if (km.labels_[idx] == cl):
				corpus_common_tags[idx].append(temp_est)

for idx,common_tags in corpus_common_tags.items():
	corpus_common_tags[idx] = frequent([tag for tag_list in common_tags for tag in tag_list])

## Write new output
output_fh = open(out_new,'w')
writer = csv.writer(output_fh, quoting = csv.QUOTE_ALL)
writer.writerow(['id','tags'])
for i in xrange(len(tags_est)):
	if type(tags_est[i]) is float:
		tags_new = corpus_common_tags[i].max()
	else:
		### Copy
		tags_new = tags_est[i][:]
		### Append the "cluster-representation" word
		if len(re.split(' ', tags_est[i])) == 1 and \
		   corpus_common_tags[i].freq(corpus_common_tags[i].max()) * corpus_common_tags[i].N() > 50 and \
		   (not corpus_common_tags[i].max() in tags_new):
			tags_new = tags_new + ' ' + corpus_common_tags[i].max()
	writer.writerow([ids[i], tags_new])

output_fh.close()

