'''
Applied (outout_5.csv --> outout_9.csv):
(1) electromagnetic --> electromagnetism
(2) blackhole / blackholes / black hole --> black-holes
(3) fourier / fourier-transformation --> fourier-transform
(4) homework --> homework-and-exercises
(5) force --> forces
(6) hamiltonian --> hamiltonian-formalism
(7) lagrangian --> lagrangian-formalism
(8) relativity --> special-relativity or general-relativity
(9) mnemonics --> mnemonic
(10) neutron --> neutrons
(11) proton --> protons
(12) coulomb --> coulombs-law
(13) gauss --> gauss-law
(14) magnitude order --> order-of-magnitude
(15) battery --> batteries
(16) home experiment --> home-experiment
(17) quantum --> quantum-mechanics
(18) mathematical --> mathematical-physics
(19) pauli --> pauli-exclusion-principle
(20) slit / slit experiment / slit-experiment --> double-slit-experiment
(21) magnetic --> magnetic-fields
(22) recommendation / recommendations --> resource-recommendations
------------------------------------------
Not applied yet:
(1) angular / angular momentum --> angular-momentum
(2) current --> electric-current
(3) representation / representations --> representation-theory (or group-representation)
(4) particle / particles --> particle-physics
(5) magnetic --> magnetic-fields
(6) group --> group-theory (or group-representation)
(7) lie --> lie-algebra
'''

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

file_old = 'output_5.csv'
fh_old = open(file_old)
reader = csv.DictReader(fh_old)
old = {}
for idx,row in enumerate(reader):
	tags = row['tags'].split()
	old[row['id']] = tags

## [check and sort all tags]
import operator
old_tags = {}
for k,v in old.items():
	for tag in v:
		old_tags[tag] = old_tags.get(tag, 0) + 1

len(old_tags)

old_tags_sorted = sorted(old_tags.items(), key=operator.itemgetter(1), reverse = True)
old_tags_sorted[0:50]
## End of [check and sort all tags]

## ===== Cheating! =====
new = {}
for idx,row in old.items():
	new_row = []
	has_fourier = False
	has_relativity = False
	has_magnitude = False
	has_home = False
	has_quantum = False
	has_pauli = False
	has_slit = False
	has_slit_experiment = False
	has_magnetic = False
	for tag in row:
		if tag == 'electromagnetic':
			new_row.append('electromagnetism')
		elif tag == 'blackhole' or tag == 'blackholes':
			new_row.append('black-holes')
		elif tag == 'fourier':
			new_row.append('fourier-transform')
			has_fourier = True
		elif tag == 'homework':
			new_row.append('homework-and-exercises')
		elif tag == 'force':
			new_row.append('forces')
		elif tag == 'hamiltonian':
			new_row.append('hamiltonian-formalism')
		elif tag == 'lagrangian':
			new_row.append('lagrangian-formalism')
		elif tag == 'relativity':
			has_relativity = True
		elif tag == 'mnemonics':
			new_row.append('mnemonic')
		elif tag == 'neutron':
			new_row.append('neutrons')
		elif tag == 'proton':
			new_row.append('protons')
		elif tag == 'coulomb':
			new_row.append('coulombs-law')
		elif tag == 'gauss':
			new_row.append('gauss-law')
		elif tag == 'magnitude':
			has_magnitude = True
			new_row.append('order-of-magnitude')
		elif tag == 'battery':
			new_row.append('batteries')
		elif tag == 'home':
			new_row.append('home-experiment')
			has_home = True
		elif tag == 'quantum':
			new_row.append('quantum-mechanics')
			has_quantum = True
		elif tag == 'mathematical':
			new_row.append('mathematical-physics')
		elif tag == 'pauli':
			new_row.append('pauli-exclusion-principle')
			has_pauli = True
		elif tag == 'slit':
			new_row.append('double-slit-experiment')
			has_slit = True
		elif tag == 'slit-experiment':
			new_row.append('double-slit-experiment')
			has_slit_experiment = True
		elif tag == 'magnetic':
			new_row.append('magnetic-fields')
			has_magnetic = True
		elif tag == 'recommendation' or tag == 'recommendations':
			new_row.append('resource-recommendations')
		else:
			new_row.append(tag)
	if has_fourier and 'transform' in new_row:
		new_row.remove('transform')
	if has_relativity:
		if not 'special' in new_row and not 'general' in new_row:
			new_row.append('special-relativity')
			new_row.append('general-relativity')
		else:
			if 'special' in new_row:
				new_row.append('special-relativity')
				new_row.remove('special')
			if 'general' in new_row:
				new_row.append('general-relativity')
				new_row.remove('general')
	if has_magnitude and 'order' in new_row:
		new_row.remove('order')
	if has_home:
		if 'experiment' in new_row:
			new_row.remove('experiment')
		if 'experiments' in new_row:
			new_row.remove('experiments')
	if has_quantum:
		if 'mechanics' in new_row:
			new_row.remove('mechanics')
		elif 'mechanic' in new_row:
			new_row.remove('mechanic')
	if has_pauli:
		if 'exclusion' in new_row:
			new_row.remove('exclusion')
		if 'principle' in new_row:
			new_row.remove('principle')
	if has_slit:
		if 'experiment' in new_row:
			new_row.remove('experiment')
		if 'experiments' in new_row:
			new_row.remove('experiments')
		if 'double' in new_row:
			new_row.remove('double')
	if has_slit_experiment:
		if 'double' in new_row:
			new_row.remove('double')
	if has_magnetic:
		if 'field' in new_row:
			new_row.remove('field')
		if 'fields' in new_row:
			new_row.remove('fields')
	new[int(idx)] = list(set(new_row))

output_name = 'output_9.csv'
output = open(output_name,'w')
writer = csv.writer(output, quoting = csv.QUOTE_ALL)
writer.writerow(['id','tags'])

keylist = new.keys()
keylist.sort()
for key in keylist:
	writer.writerow([key,' '.join(new[key])])

output.close()

## TODO: String distance?
# from nltk import edit_distance
# from leven import levenshtein
# from sklearn.cluster import dbscan

# data = ["ACCTCCTAGAAG", "ACCTACTAGAAGTT", "GAATATTAGGCCGA"]
# data = [k for k,v in old_tags_sorted[0:50]]

# def lev_metric(x, y):
# 	i, j = int(x[0]), int(y[0]) # extract indices
# 	return levenshtein(data[i], data[j])

# X = np.arange(len(data)).reshape(-1, 1)
# X
# dbscan(X, metric=lev_metric, eps=5, min_samples=2)







