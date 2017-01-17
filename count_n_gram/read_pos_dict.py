import csv

### Read pos_tagging result
pos_fh = open('pos_dict.csv')
reader = csv.DictReader(pos_fh)
pos_dict = {}
for idx,row in enumerate(reader):
	pos = row['POS'].split()
	pos_dict[row['word']] = {pos_count.split(':')[0]:int(pos_count.split(':')[1]) for pos_count in pos}

pos_fh.close()

print 'POS tag for "electromagnetic":', pos_dict['electromagnetic']
print 'POS tag for "electromagnetism":', pos_dict['electromagnetism']

### Calculate the noun-ratio for each word
### (the proportion of time this word being tagged as a noun)
noun_ratio = {}
for word in pos_dict:
	noun_count = 0
	other_count = 0
	for pos in pos_dict[word]:
		if pos in ['NN', 'NNP', 'NNPS', 'NNS']:
			noun_count = noun_count + pos_dict[word][pos]
		else:
			other_count = other_count + pos_dict[word][pos]
	noun_ratio[word] = float(noun_count) / (noun_count + other_count)

print 'The noun ratio of "electromagnetic":', noun_ratio['electromagnetic']
print 'The noun ratio of "electromagnetism":', noun_ratio['electromagnetism']

