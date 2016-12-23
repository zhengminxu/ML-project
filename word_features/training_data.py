# -*- coding: utf-8 -*-

import os
import pandas as pd
import re
import numpy as np
import time
from pandas import DataFrame
import nltk

starttime1=time.time()
print 'loading data...'

#%%
stopwords_txt=open('your_file_location/stopword.txt', 'r').read()
stopwords_split = stopwords_txt.split( )
punctuation='!,.:;?'

#%%
def remove_stopword(s):
    con=s.lower().split()
    l=[w for w in con if w not in stopwords_split]
    return l


def remove_punctuation(s):
    s = s.lower()
    s = re.sub('\n', ' ', s)
    s = re.sub('<\S+>', ' ', s)
    for c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“':
        s = s.replace(c, ' ')
    return s


def remove_html(s):
    cleaner=re.sub('<.*?>', ' ', s)
    return cleaner


def clean_all(s):
    s=s.lower()
    s=remove_html(s)
    s=remove_punctuation(s)
    return(s)

#%%
train_dir = r'your_file_location/train_data'
train_file = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
test_file = r'your_file_location/test.csv'
#%%
train_data = []
all_data = []
for index, f in enumerate(train_file):
    train_data.append(pd.read_csv(f))
    all_data.append(pd.read_csv(f))
#%%
test_data = pd.read_csv(test_file)
all_data.append(test_data)


def term_ferq(l):
    tf={}
    for w in l:
        if w in tf:
            tf[w] += 1
        else:
            tf[w] = 1
    for key in tf:
        tf[key]=float(tf[key])/float(len(l))
    return tf


def iverse_doc_freq(data):
    idf={}
    D=0
    for d in data:
        for ss in d.iloc[:,[1,2]].get_values():
            title=nltk.word_tokenize(clean_all(ss[0]))
            context=nltk.word_tokenize(clean_all(ss[1]))
            s=title+context
            D+=1
            for term in set(s):
                if term in idf:
                    idf[term] += 1
                else:
                    idf[term] = 1
    for key in idf:
        idf[key]=np.log(float(D)/float(idf[key]))
    return idf


def is_in_l(w,l):
    if w in set(l):
        label=1
    else:label=0
    return label


def position(indx,title_len,context_len):
    if indx<=title_len:
        position=float(indx)/float(title_len)
    else:
        position=float(indx-title_len)/float(context_len)
    return position


def token_tag(ss):
    tokens=nltk.word_tokenize(ss)
    tagged=nltk.pos_tag(tokens)
    return tokens,tagged

print 'Data loaded ; Operation time:{} minute'.format((time.time()-starttime1)/60)

#%%
print 'Building training data...'

idf=iverse_doc_freq(all_data)

tag_to_int={}
tag_number=1

with open('training_data_output.csv', 'w+') as f:
    f.write("'term_ferquency','inverse_doc_frequency','positions','in_title','in_context','pos_tag','in_label'\n")
    for i,data in enumerate(train_data):
        starttime=time.time()
        for ss in data.iloc[:,[1,2,3]].get_values():
            title,title_pos_tag=token_tag(clean_all(ss[0]))
            title_len=len(title)
            context,context_pos_tag=token_tag(clean_all(ss[1]))
            context_len=len(context)
            tag=ss[2].lower().split()
            l=title+context
            word_tag=title_pos_tag+context_pos_tag
            tf=term_ferq(l)
            term_ferquency=[]
            inverse_doc_frequency=[]
            positions=[]
            in_title=[]
            in_context=[]
            in_label=[]
            pos_tag=[]
            for index,wordtag in enumerate(word_tag):
                if not wordtag[0] in stopwords_split and not wordtag[0] in punctuation:
                    if not wordtag[1] in tag_to_int:
                        tag_to_int[wordtag[1]]=tag_number
                        tag_number+=1
                    indx=index+1
                    term_ferquency.append(tf[wordtag[0]])
                    inverse_doc_frequency.append(idf[wordtag[0]])
                    positions.append(position(indx,title_len,context_len))
                    in_title.append(is_in_l(wordtag[0],title))
                    in_context.append(is_in_l(wordtag[0],context))
                    in_label.append(is_in_l(wordtag[0],tag))
                    pos_tag.append(tag_to_int[wordtag[1]])

            table=[term_ferquency,inverse_doc_frequency,positions,in_title,in_context,pos_tag,in_label]
            df=DataFrame(table)
            df=df.transpose()
            cols=['term_ferquency','inverse_doc_frequency','positions','in_title','in_context','pos_tag','in_label']
            df.columns=cols
            df.to_csv(f,mode='a',index=False,header=False)
        print 'Data builed : {}/6 ; Operation time : {:04.2f} minute'.format(i+1,(time.time()-starttime)/60)
    f.close()
print 'Training data saved : training_data.csv'
