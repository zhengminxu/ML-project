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

## 方法2: find general tagset
- 步驟：
（1）首先我們想要利用NN預測出所有在文章中的字，哪些是tag而那些不是，此為輸入是word，output是0或1的regression task。 
（2）接著我們將決定如何將一個word embed成向量，我們採取的作法是，看這個字在文章中出現幾次，如此一來就有一個維度為文章總數的向量，藉此來表示一個字。
（3）再來利用其他topic的data train我們的model，用robotic作為validation set，隨機train 500次並存取acc最高的model(best.h5)。
（4）利用train好的model去預測physics類別中的字，得到一個general的tag set。
（5）對此tag set做篩選，刪掉出現次數較少的，並且把這幾個字搜尋其前後出現的字，若有出現次數很多的，就將其組成2-gram加入tag set中。
- 討論：
此方法雖無法自行預測出此篇文章的tag，但可以用於輔助其他較heuristic的方法。不過預測出來的字過於general，似乎成效普通。
## 方法3
todo

# 結論
todo
