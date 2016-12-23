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

if (len(sys.argv) < 2):
	file_ans = 'answer.csv'
	file_est = 'test_string_theory_est.csv'
elif (len(sys.argv) == 2):
	file_name = sys.argv[1]
	file_ans = file_name.split('.')[0] + '_ans.csv'
	file_est = file_name.split('.')[0] + '_est.csv'


fh_ans = open(file_ans)
reader = csv.DictReader(fh_ans)
ans = {}
for idx,row in enumerate(reader):
	tags = row['tags'].split()
	ans[row['id']] = tags

fh_out = open(file_est)
reader = csv.DictReader(fh_out)
out = {}
for idx,row in enumerate(reader):
	tags = row['tags'].split()
	out[row['id']] = tags

## Precision
total_guess_tags = 0
precision_tags = 0
for key in out:
	for tag in out[key]:
		total_guess_tags = total_guess_tags + 1
		if tag in ans[key]:
			# print '"%s":"%s"' % (key, tag)
			precision_tags = precision_tags + 1

precision = precision_tags / float(total_guess_tags)
print 'Precision = %f' % precision

## Recall
total_true_tags = 0
recall_tags = 0
short_true_tags = {}
for key in ans:
	for tag in ans[key]:
		if len(tag) < 4:
			short_true_tags[tag] = short_true_tags.get(tag, 0) + 1
		total_true_tags = total_true_tags + 1
		if tag in out[key]:
			# print '"%s":"%s"' % (key, tag)
			recall_tags = recall_tags + 1

recall = recall_tags / float(total_true_tags)
print 'Recall = %f' % recall

f1_score = 2 / (1/precision + 1/recall)
print 'F1 score = %f' % f1_score



