# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk import FreqDist
import nltk
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import re
import sys

file_name = sys.argv[1]
output_name = file_name.split('.')[0] + '_est.csv'
# file_name = './test_string_theory.csv'
# file_name = './test.csv'
# output_name = 'output_4.csv'
# output_name = 'output_test_best.csv'

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
# df = open('./test.csv')

# my_stopwords = set([u'using', u'use', u'file', u'files', u'get', u'way', u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'with', u'had', u'should', u'to', u'only', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'did', u'these', u't', u'each', u'where', u'because', u'doing', u'theirs', u'some', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'below', u'does', u'above', u'between', u'she', u'be', u'we', u'after', u'here', u'hers', u'by', u'on', u'about', u'of', u'against', u's', u'or', u'own', u'into', u'yourself', u'down', u'your', u'from', u'her', u'whom', u'there', u'been', u'few', u'too', u'themselves', u'was', u'until', u'more', u'himself', u'that', u'but', u'off', u'herself', u'than', u'those', u'he', u'me', u'myself', u'this', u'up', u'will', u'while', u'can', u'were', u'my', u'and', u'then', u'is', u'in', u'am', u'it', u'an', u'as', u'itself', u'at', u'have', u'further', u'their', u'if', u'again', u'no', u'when', u'same', u'any', u'how', u'other', u'which', u'you', u'who', u'most', u'such', u'why', u'a', u'don', u'i', u'having', u'so', u'the', u'yours', u'once'])
meaning_less = ['p','would','could','via','emp','two','must','make',
                'e','c','using','r','vs','versa','based','three']
my_stopwords = set(stopwords.words('english')).union(meaning_less)
import string
punct = set(string.punctuation)

def clear_stopwords(context):
	letters = re.sub("[^a-zA-Z]", " ", context).lower().split()
	# stopword = set(stopwords.words('english'))
	clear = [c for c in letters if c not in my_stopwords]
	return clear

def find_all_dashed(context):
	dashed = re.findall("[a-zA-Z]+\\-[a-zA-Z]+", context, flags=0)
	return dashed

def find_all_dashed2(context):
	dashed = re.findall("[a-zA-Z]+\\-[a-zA-Z]+\\-[a-zA-Z]+", context, flags=0)
	return dashed

# clear_stopwords('What is your simplest explanation of the string theory?')
# clear_stopwords_keep_dash('What is your simplest explanation of the string-theory?')

from nltk import ngrams
def make_n_gram(context, n):
	letters = re.sub("[^a-zA-Z]", " ", context).lower().split()
	clear = [c for c in letters if c not in my_stopwords]
	context = ' '.join(word for word in clear)
	# print context
	## Remove punctuation and make it all lowercase
	context = ''.join(ch for ch in context if ch not in punct)
	n_grams = ngrams(context.split(), n)
	return ['-'.join(g) for g in n_grams]

# make_n_gram('What is your simplest explanation of the string theory?', 2)

# title2 = ''.join(ch for ch in sentence2.lower() if ch not in exclude)
# 	n_grams = ngrams(title2.split(), n)
# 	for grams in n_grams:
# 		title.append("-".join(grams))

def remove_html(context):
	## remove the content between <code> and </code> first
	# cleaner = re.compile('<code>.*?</code>')
	# context = re.sub(cleaner,'',context)
	cleaner = re.compile('<.*?>')
	clean_text = re.sub(cleaner,'',context)
	return clean_text

test_string = '''<p>This is a question that has been posted at many different forums, I thought maybe someone here would have a better or more conceptual answer than I have seen before:</p>

<p>Why do physicists care about representations of Lie groups? For myself, when I think about a representation that means there is some sort of group acting on a vector space, what is the vector space that this Lie group is acting on? </p>

<p>Or is it that certain things have to be invariant under a group action?
maybe this is a dumb question, but i thought it might be a good start...</p>

<p>To clarify, I am specifically thinking of the symmetry groups that people think about in relation to the standard model. I do not care why it might be a certain group, but more how we see the group acting, what is it acting on? etc.</p>'''
# remove_html(re.sub("\\n", " ", test_string))
# remove_html('<p>How do you go about it if the light beam has different polarizations in different parts of the transverse plane? One example is a <a href=""http://en.wikipedia.org/wiki/Radial_polarization"">radially polarized</a> beam. More generally, is there a good technique for sampling the local polarization (which might be linear, elliptical, or circular, anywhere on the <a href=""http://en.wikipedia.org/wiki/Poincare_sphere"">Poincaré sphere</a>) in one transverse plane?</p>')

# def remove_code(context):
# 	cleaner = re.compile('<code>.*?</code>')
# 	clean_text = re.sub(cleaner,'',context)
# 	return clean_text

# remove_code('<pre><code>aW1 := c[z, 0] == If[z &gt; 0, 0, Ca]    rW2 := d Derivative[1, 0][c][0, t] - v c[0, t] == v Ca</code></pre>')

def frequent(context):
	freq = FreqDist(context)
	return freq

## 2016/12/22: POS tagging
# temp_context = 'Velocity of Object from electromagnetic field'
# nltk.pos_tag(re.sub("[^a-zA-Z]", " ", temp_context).lower().split())
# nltk.pos_tag(clear_stopwords(temp_context))
# temp_context = 'What is your simplest explanation of the string theory?'
# nltk.pos_tag(re.sub("[^a-zA-Z]", " ", temp_context).lower().split())
# nltk.pos_tag(clear_stopwords(temp_context))
# temp_context = 'What is your simplest explanation of the string theories?'
# nltk.pos_tag(re.sub("[^a-zA-Z-]", " ", temp_context).lower().split())
# nltk.pos_tag(clear_stopwords(temp_context))
## ==> we should not clear stopwords before POS tagging

# from nltk import ngrams
# sentence = 'this is a foo bar sentences and i want to ngramize it'
# n = 2
# n_grams = ngrams(sentence.split(), n)
# for grams in n_grams:
# 	print grams




df = open(file_name)
reader_for_all = csv.DictReader(df)
all_words = []
all_dashed = []
all_dashed2 = []
for idx,row in enumerate(reader_for_all):
	# print row['id']
	## (1) all_words
	title = clear_stopwords(row['title']) ## return list
	all_words.append([t for t in title])
	content = remove_html(row['content'])
	content = clear_stopwords(content)
	all_words.append([t for t in content])
	## (2) all_dashed
	title_keep_dash = find_all_dashed(row['title']) ## return list
	# print title_keep_dash
	if len(title_keep_dash):
		all_dashed.append([t for t in title_keep_dash])
	content = remove_html(row['content'])
	content_keep_dash = find_all_dashed(content)
	# print content_keep_dash
	if len(content_keep_dash):
		all_dashed.append([t for t in content_keep_dash])
	## (3) all_dashed2
	title_keep_dash2 = find_all_dashed2(row['title']) ## return list
	# print title_keep_dash
	if len(title_keep_dash2):
		all_dashed2.append([t for t in title_keep_dash2])
	content_keep_dash2 = find_all_dashed2(content)
	# print content_keep_dash
	if len(content_keep_dash2):
		all_dashed2.append([t for t in content_keep_dash2])
	## (4) POS tagging
	# title_pos = nltk.pos_tag(re.sub("[^a-zA-Z]", " ", remove_html(re.sub("\\n", " ", row['content']))).lower().split())
	# content_pos = nltk.pos_tag(re.sub("[^a-zA-Z]", " ", remove_html(re.sub("\\n", " ", row['content']))).lower().split())
	# 	print letters

# all_words = set([w.lower() for w_list in all_words for w in w_list])
all_words = frequent([w.lower() for w_list in all_words for w in w_list])
# print all_words.most_common(100)
# print all_words['spacetime']
# print '===================\n'
all_dashed = frequent([w.lower() for w_list in all_dashed for w in w_list])
# print all_dashed.most_common(100)
# print all_dashed['space-time']
# for k in all_dashed:
# 	if len(k) < 4:
# 		print k
all_dashed2 = frequent([w.lower() for w_list in all_dashed2 for w in w_list])
# print all_dashed2.most_common(100)
short_2gram = {}
short_tags = {}
df.close()

df = open(file_name)
reader = csv.DictReader(df)
preds = defaultdict(list)
output = open(output_name,'w')
writer = csv.writer(output, quoting = csv.QUOTE_ALL)
writer.writerow(['id','tags'])
# count = 0
for idx,row in enumerate(reader):
	# print '\n'
	# count = count + 1
	# if count > 2:
		# break
	# print idx
	# print row
	title = clear_stopwords(row['title']) ## return list
	# title = title + make_n_gram(row['title'], 2)
	# print title
	content = remove_html(row['content'])
	content = clear_stopwords(content)
	# content = content + make_n_gram(row['content'], 2)
	# if row['id'] == '37':
	# 	print row['content']
	# 	print '===================\n'
	# 	content = remove_html(row['content'])
	# 	content = clear_stopwords(content)
	# 	print content
	# 	print '===================\n'
	# 	letters = re.sub("[^a-zA-Z-]", " ", remove_html(re.sub("\\n", " ", row['content']))).lower().split()
	# 	print letters
	# 	print '===================\n'
	# 	clear = [c for c in letters if c not in my_stopwords]
	# 	print clear
	# 	print '===================\n'
	# 	pos_tagging = nltk.pos_tag(letters)
	# 	print pos_tagging
	# 	print '===================\n'
	# print content
	freq_title = frequent(title)
	# print freq_title.most_common(5)
	freq_content = frequent(content)
	# print freq_content.most_common(5)
	all_grams = make_n_gram(row['title'], 2) + \
	            make_n_gram(remove_html(row['content']), 2)
	all_grams2 = make_n_gram(row['title'], 3) + \
	             make_n_gram(remove_html(row['content']), 3)
	print 'ID = %s' % row['id']
	# print row['content']
	# print remove_html(row['content'])
	freq_grams = frequent(all_grams)
	print freq_grams.most_common(3)
	freq_grams2 = frequent(all_grams2)
	print freq_grams2.most_common(3)
	# if row['id'] == '185':
		# print freq_grams.most_common(5)
	preds[row['id']].append(' '.join(title[:]))
	# if row['id'] == '73':
		# print 'preds:'
		# print preds
	#writer.writerow([row['id'],' '.join(title[:3])])
	common = set(content).intersection(title)
	# print common
	# common = common.union([k for (k,v) in freq_grams.most_common(1) if v > 1])
	freq_grams_v = freq_grams.most_common(1)[0][1]
	freq_grams_max = [k for (k,v) in freq_grams.items() if v == freq_grams_v]
	print freq_grams_max
	common2 = [k for (k,v) in freq_grams.most_common(1) if v > 1 and 'http' not in k and (k in all_dashed or k in all_dashed2)]
	# common2 = [k for (k,v) in freq_grams.most_common(1) if v > 2 and 'http' not in k]
	# print common2
	if len(common2):
		common2_copy = [x for x in common2]
		for dashed in common2_copy:
			if len(dashed) < 4:
				short_2gram[dashed] = short_2gram.get(dashed, 0) + 1
				common2.remove(dashed)
			elif all_words[re.sub('-','',dashed)] > all_dashed[dashed]:
				common2.remove(dashed)
				common2.append(re.sub('-','',dashed))
		common3 = set([word for word in common for word2 in common2 if not word in word2])
	else:
		common3 = common
	# print common3
	common = common3.union(common2)
	# print row['id']
	# print common
	common_copy = [x for x in common]
	for tag in common_copy:
		if len(tag) < 3:
			short_tags[tag] = short_tags.get(tag, 0) + 1
			common.remove(tag)
	# print common
	# print '============'
	# print '\n'
	temp = []
	if len(common) == 0:
		for t in title:
			if t not in meaning_less:
				temp.append(t)
		#print('ID : {} , Title : {}'.format(idx+1,title))
		writer.writerow([row['id'],' '.join(temp)])
	else:
		writer.writerow([row['id'],' '.join(common)])
		#writer.writerow([row['id'],' '.join(set(content).intersection(title))])

df.close()
output.close()
