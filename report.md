# 主題
- Transfer Learning on StackExchange Tags
# 組員
- 許正忞，林家慶，陳力維，王冠驊

# 方法
## 方法1: doc2vec & clustering
- 步驟：
（1）對test.csv中的文章進行前處理，包括tokenize、stem、去除html標記、去除stopwords等。
（2）用處理過的文章建立200維的doc2vec模型。
（3）用Kmeans把所有文章的doc2vec分為20群，對每一群建立一個名詞字典。
（4）根據每篇文章被分到哪一群，計算出這篇文章與這一群的名詞字典中最相似的5個詞，把它們作為tag。
- performance: 0.02
- 討論：
這個方法的效果不是很好，一方面可能因為doc2vec對文章的feature把握不是很準，另一方面可能也因為分群數量太少。另外，這個方法也沒有考慮到2-grams及3-grams，而實際上很多真正的tag都是二連字和三連字。這些問題將在其他方法中得到解決。

## 方法2
todo
## 方法3
todo

# 結論
todo