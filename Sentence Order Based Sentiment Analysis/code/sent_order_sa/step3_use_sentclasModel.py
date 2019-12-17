# -*- coding: utf-8 -*-

import os
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
import types
import os
from scipy.sparse import *
from scipy import *
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn import neighbors
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
from sklearn.cross_validation import cross_val_score

def get_textX_and_y_seggedFile(segged_filename, input_sep, segged_cont_idx, segged_sep=' ', 
                                did_idx=None, label_idx=None):
        textX = []
        dids = []
        labels = []
        for line in open(segged_filename):
            line = line.strip('\n')
            if input_sep != None:
                segged_content = line.split(input_sep)[segged_cont_idx]
            else:
                segged_content = line
            if did_idx!=None:
                dids.append(line.split(input_sep)[did_idx])
            else:
                dids.append('-1')
            if label_idx!=None:
                labels.append(int(line.split(input_sep)[label_idx]))
            else:
                labels.append(-1)                       
            if cmp(segged_sep,' ')!=0: segged_content = segged_content.replace(segged_sep, ' ')
            textX.append(segged_content)
        return textX,dids, labels

train_textX, train_dids, train_labels = get_textX_and_y_seggedFile('segged_train.txt','\t',-1,did_idx=0,label_idx=1)

sentclas_clf = Pipeline([('vect', CountVectorizer(max_features = 9000,ngram_range=(1,2))), ('tfidf', TfidfTransformer()), 
                     ('clf', SVC(C=0.99, kernel = 'linear', probability=True))]) 

sentclas_clf.fit(train_textX, train_labels)

test_textX, test_dids, test_labels = get_textX_and_y_seggedFile('segged_test.txt','\t',-1,did_idx=0)
pred = sentclas_clf.predict(test_textX)
pred_prob = sentclas_clf.predict_proba(test_textX)
i=0
result_file= open('segged_test_sentpred.txt','w')
for line in open('segged_test.txt'):
    line= line.strip()
    if len(line)<2:continue
    result_file.write(line + '\t' + str(pred[i])+'\t' + str(pred_prob[i]) + '\n')
    i = i+1



train_sen_textX, train_sen_dids, train_sen_labels = get_textX_and_y_seggedFile('train_senpos.txt','\t',-1,did_idx=0)

pred = sentclas_clf.predict(train_sen_textX)
pred_prob = sentclas_clf.predict_proba(train_sen_textX)
i=0
result_file= open('train_senpos_sentpred.txt','w')
for line in open('train_senpos.txt'):
    line= line.strip()
    if len(line)<2:continue
    result_file.write(line + '\t' + str(pred[i])+'\t' + str(pred_prob[i]) + '\n')
    i = i+1
    
test_sen_textX, test_sen_dids, test_sen_labels = get_textX_and_y_seggedFile('test_senpos.txt','\t',-1,did_idx=0)

pred = sentclas_clf.predict(test_sen_textX)
pred_prob = sentclas_clf.predict_proba(test_sen_textX)
i=0
result_file= open('test_senpos_sentpred.txt','w')
for line in open('test_senpos.txt'):
    line= line.strip()
    if len(line)<2:continue
    result_file.write(line + '\t' + str(pred[i])+'\t' + str(pred_prob[i]) + '\n')
    i = i+1


