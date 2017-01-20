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
        return [1,0] #not label
    else:return [0,1] #is label

def to_word(d):
    liste=d.iloc[:,7].values.tolist()
    return liste

tra_data = pd.read_csv(r'training_data.csv')
tes_data = pd.read_csv(r'testing_data.csv')
test_data=[]

training_data=tra_data.iloc[:,[0,1,2,3,4]].values.tolist()
#print training_data[0] #[[0.0147058823529, 8.6699201982, 0.235294117647, 1.0, 0.0], []...]
label=map(label_to_list,tra_data.iloc[:,6].values.tolist())
#print label #[[1, 0], [0, 1], [1, 0]...]

_ids=[]
test_data=[]
for _id,group in tes_data.groupby('\'id\'',as_index=False,group_keys=False,sort=False):
    #print _id
    _ids.append(_id)
    #print group
    test_data.append(group.iloc[:,[1,2,3,4,5]].values.tolist())
#print _ids
words=[x for x in tes_data.groupby('\'id\'',as_index=False,group_keys=False,sort=False).apply(to_word)]
#print words[:10] #[['spin', 'relates', 'subatomic', 'particles', 'hear'...],[...],...]


word_feature_len=5

x=np.array(training_data)
y=np.array(label)
#print type(x),x.shape,y.shape #<type 'numpy.ndarray'> (4645965, 5) (4645965, 2)

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
with open('ans.csv','w')as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    spamwriter.writerow(['id','tags'])
    for i in range(len(test_data)):
        #print words[i][6]
        txt=np.array(test_data[i])
        #print txt.shape
        result=model.predict(txt)
        #print result[:10]
        #print type(result)
        tags=[]
        #maxx=result.argmax(axis=0)[0]
        #print maxx,type(maxx)
        for j in range(len(result)):
            if result[j][0]>0.07:
                tags.append(words[i][j])
        tags=list(set(tags))
        print tags
        spamwriter.writerow([_ids[i],' '.join(tags)])
    csvfile.close()
