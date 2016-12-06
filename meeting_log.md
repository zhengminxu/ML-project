# Simple method
- beyond simple baseline: 0.08105
	+ use text.csv only
	+ find the most (3) frequent nouns/proper nouns from the [title + body] of each question

## Xu
- Goal: Find most similar words to a given question
    + most similar words: nouns/phrases throughout the whole doc
- method:
    + Use doc2vec to cluster questions (number of clusters)
    + Find nouns/phrases in each cluster
    + For a given question, first assign it to a cluster, then calculate 3 most similar words/phrases to it in that cluster


## Lin
- Find candidate: 所有问题中出现次数最多的前（200）个词
- 找一些feature，二元分类
- probability过threshold

## Wang
1) 对每篇文章，求出每个词的tfidf，取最大的几个
2) 从doc2vec找附近的word2vec (train过的word2vec)
3) CNN

## Chen
- NN

* 不一定所有给的文件都用到
