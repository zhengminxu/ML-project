import os
import pandas as pd
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1l2
from keras.models import load_model

def label_to_list(x):
    if x==0:
        return [1,0]
    else:return [0,1] 

def to_word(d):
    liste=d.iloc[:,7].values.tolist()
    return liste

tra_data = pd.read_csv(r'training_data.csv')
tes_data = pd.read_csv(r'testing_data.csv')
test_data=[]

training_data=tra_data.iloc[:,[0,1,2,3,4]].values.tolist()
label=map(label_to_list,tra_data.iloc[:,6].values.tolist())

_ids=[]
test_data=[]
for _id,group in tes_data.groupby('\'id\'',as_index=False,group_keys=False,sort=False):
    _ids.append(_id)
    test_data.append(group.iloc[:,[1,2,3,4,5]].values.tolist())
words=[x for x in tes_data.groupby('\'id\'',as_index=False,group_keys=False,sort=False).apply(to_word)]


word_feature_len=5

x=np.array(training_data)
y=np.array(label)

batch_size=50
epoch=2

reg=l1l2(l1=0.01,l2=0.01)
model=Sequential()
model.add(Dense(2,activation='sigmoid',W_regularizer=reg,input_dim=x.shape[1]))
model.compile(optimizer='rmsprop',loss='binary_crossentropy')
if os.path.isfile('my_model')==False:
    model.save('my_model')
model=load_model('my_model')
model.fit(x,y,batch_size=batch_size,nb_epoch=epoch,shuffle=True)
model.save('my_model')
print "model saved"

model=load_model('my_model')
with open('output.csv','w')as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    spamwriter.writerow(['id','tags'])
    for i in range(len(test_data)):
        txt=np.array(test_data[i])
        result=model.predict(txt)
        tags=[]
        for j in range(len(result)):
            if result[j][0]>0.07:
                tags.append(words[i][j])
        tags=list(set(tags))
        spamwriter.writerow([_ids[i],'\"'+' '.join(tags)+'\"'])
    csvfile.close()
