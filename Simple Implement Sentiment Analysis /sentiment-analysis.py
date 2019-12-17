# -*- coding: utf-8 -*-
"""
Created on Tue May 09 23:58:40 2017

"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import jieba
import jieba.analyse
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB  
from sklearn.ensemble import RandomForestClassifier  
import jieba.posseg as pseg
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#read data
fdir = ""
fdir = unicode(fdir , "utf-8")

train = pd.read_table(fdir + "train.txt", header = None)
test = pd.read_table(fdir + "test.txt", header = None)

#change labels into numrical variables 
label = {"利好":1, "中性":2, "利空":0}
train[2] = [label[i] for i in train[2]]

#rename
train.columns = ['id', 'text', 'label']
test.columns = ['id', 'text']

#"text" and "label" columns 
traintext = train["text"]
label = train["label"]
testtext = test["text"]

#remove punctuations 
traintext = [stri.decode("utf8") for stri in traintext]  
traintext = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"),
             "".decode("utf8"), stri) for stri in traintext]  

testtext = [stri.decode("utf8") for stri in testtext]  
testtext = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"),
             "".decode("utf8"), stri) for stri in testtext] 

#retain Chinese
train_text=[]
for i in range(len(traintext)):
    train_text.append(''.join(re.findall(u'[\u4e00-\u9fff]+', traintext[i])))
    
test_text=[]
for i in range(len(testtext)):
    test_text.append(''.join(re.findall(u'[\u4e00-\u9fff]+', testtext[i])))

#segword
train_words = [jieba.lcut(i) for i in train_text]
test_words = [jieba.lcut(i) for i in test_text]

#filter stopwords
stoplist = {}.fromkeys([ line.strip() for line in open(fdir + "stopwords.txt") ])  

train_segwords=[]
for word_list in train_words:  
    word_list = [word for word in word_list if word.encode('utf-8') not in stoplist] 
    train_segwords.append(word_list)
    
test_segwords=[]
for word_list in test_words:  
    word_list = [word for word in word_list if word.encode('utf-8') not in stoplist] 
    test_segwords.append(word_list)

#matrix word frequencyy
trainset = [" ".join(words_list) for words_list in train_segwords]
testset = [" ".join(words_list) for words_list in test_segwords]

#model 
def voting_classify():
    
    clf1 = SVC(C=0.99, kernel = 'linear', probability=True)
    clf2 = RandomForestClassifier(random_state=0, n_estimators=200)
    #clf3 = LogisticRegression(random_state=1)
    #clf4 = MultinomialNB(alpha = 0.1)
    #clf5 = xgboost.XGBClassifier()
    clf = VotingClassifier(estimators=[
        ('svc',clf1),
        ('rf',clf2),
        #('lr',clf3),
        #('NB',clf4),
    ],
        voting='soft'
    )
    return clf

text_clf = Pipeline([('vect', CountVectorizer(max_features = 9000,ngram_range=(1,2))), ('tfidf', TfidfTransformer()), 
                     ('clf', voting_classify())]) 
text_clf = text_clf.fit(trainset, label)
predict = text_clf.predict(testset)

label = {1:"利好", 2:"中性", 0:"利空"}
predict = [label[i] for i in predict]
id = list(test['id'])

predict = pd.DataFrame(predict)
id = pd.DataFrame(id)

result = pd.concat([id, predict], axis=1)
result.columns = ['id', 'text']


outputfile = fdir + "prediction-investopinion-stacking.txt"
result.to_csv(outputfile, index = False, header = False, encoding = 'utf-8')

res = pd.DataFrame(pred_array)
outputfile = fdir + "pred_array.txt"
res.to_csv(outputfile, index = False, header = False, encoding = 'utf-8')