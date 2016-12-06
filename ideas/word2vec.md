# Simple method
- beyond simple baseline: 0.08105
	+ use text.csv only
	+ find the most (3) frequent nouns/proper nouns from the [title + body] of each question

# Other
- Goal: Find most similar words to a given question
    + most similar words: nouns/phrases throughout the whole doc
- method:
    + Use doc2vec to cluster questions (number of clusters)
    + Find nouns/phrases in each cluster
    + For a given question, first assign it to a cluster, then calculate 3 most similar words/phrases to it in that cluster