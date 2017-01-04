from sklearn.preprocessing import Normalizer
from keras.models import load_model
from scipy import sparse
from ast import literal_eval
import numpy as np
import sys #argv[1] is the path containing all the data
import gc
embed_SIZE = 700
mul = 10
with open('../feat/test_feats', 'r') as f:
	temp, x_test = literal_eval(f.readlines()[0])

ll = lambda x: float(len(x))
lt = map(ll, temp)
print len(temp)
qq = 0
tags = np.array(x_test)
for w in x_test:
	if not qq:
		z = np.array([doc.count(w) * mul / lt[i] for i, doc in enumerate(temp)])
		qq = 1
	else:
		z = np.vstack((z, np.array([doc.count(w) * mul / lt[i] for i, doc in enumerate(temp)])))
x_test = z
print x_test.shape
gc.collect()
print "Doing LSA"
print "SVD...."
u, s, v = sparse.linalg.svds(x_test, embed_SIZE)
n = Normalizer(copy=False)
x_test = n.fit_transform(u * s.transpose())
np.save('../feat/svd.npy', x_test)

print "loading Model..."
model = load_model('../model/best.h5')
print "Getting test..."
n_large = 200
p = model.predict(x_test, batch_size=64)
p = np.array([x[0] for x in p])
out = np.argpartition(p, -1 * n_large)[-1 * n_large:]
out = out[np.argsort(p[out])[::-1]]
t = tags[out]
print p
print t

