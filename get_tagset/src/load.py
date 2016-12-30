import numpy as np
import cPickle as cp
import sys
import gc
from gensim.models import Word2Vec
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam

wmodel_path = "../model/"
vec_path = '../feat/'

def load_data():
	gc.disable()
	with open(vec_path + sys.argv[2], 'rb') as f:
		print "x"
		x_train, x_val, y_train, y_val = cp.load(f)
		print "x"
		gc.enable()
	output_len = len(y_train[0])
	#EOF at end
	print "Maximun output sequences : ", output_len - 1
	return x_train, x_val, y_train, y_val, output_len

def embed_content(maxlen):
	#In = Input(shape=(None, 500))
	model = Sequential()
	model.add(LSTM(256, input_shape=(None, 500), return_sequences=True))
	model.add(LSTM(128))
	model.add(Dense(64, activation='relu'))
	#decoder
	model.add(RepeatVector(maxlen))
	model.add(LSTM(256, return_sequences=True))
	model.add(TimeDistributedDense(500, activation='softmax'))
	return model
	# under cosine similarity, fine to use softmax
	return Model(input=In, output=decoder)

def train_model():
	x_train, x_val, y_train, y_val, output_len = load_data()
	model = embed_content(output_len)
	model.compile(loss=cosine_proximity, optimizer=Adam(lr=0.01), metrics=['accuracy'])
	model.fit(x_train, y_train, batch_size=32, nb_epoch=10, validation_data=(x_val, y_val))
def get_result(model, y_pred):
	model = Word2Vec.load(wmodel_path + sys.argv[1])
	model.most_similar(positive=[y_pred])
train_model()
	
