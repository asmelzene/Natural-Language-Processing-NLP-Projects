# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 12:38:07 2017

@author: liyongchun
"""

from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression   
from string import *
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

train_data = []
train_lab = []
test_data = []
for line in open('train_senpos_sentpred.txt'):
    line = line.strip()
    if len(line)<3:continue
    [did,senposVec, true_label, content, pred_label, pred_prob] = line.split('\t')
    senposVec = senposVec.replace('[','')
    senposVec = senposVec.replace(']','')
    x = [float(item) for item in senposVec.split(', ')]
    if cmp(true_label, pred_label)==0:
        y = 1
    else:
        y=0
    train_data.append(x)
    train_lab.append(y)
    
test_senpos_sentpred = pd.read_table('test_senpos_sentpred.txt')

for line in open('test_senpos_sentpred.txt'):
    line = line.strip()
    if len(line)<3:continue
    [did,senposVec, content, pred_label, pred_prob] = line.split('\t')
    senposVec = senposVec.replace('[','')
    senposVec = senposVec.replace(']','')
    x = [float(item) for item in senposVec.split(', ')]
    test_data.append(x)

train_X = np.array(train_data)
train_y = np.array(train_lab)
test_X = np.array(test_data)
#clf = svm.SVC(kernel='linear', probability=True) 
clf = RandomForestClassifier(random_state=0, n_estimators=200) 

#wight of sentence order
#def voting_classify():
    
#    clf1 = SVC(C=0.99, kernel = 'linear', probability=True)
#    clf2 = RandomForestClassifier(random_state=0, n_estimators=200)
#    clf = VotingClassifier(estimators=[('svc',clf1),('rf',clf2)],voting='soft')
#    return clf

#clf = Pipeline([('clf', voting_classify())]) 
clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)
pred_prob = clf.predict_proba(test_X)
predict_weight = pred_y
predict_weight_prob = pred_prob

dindex = []
predict_label = []
predict_prob = []
test_text = []
for line in open('test_senpos_sentpred.txt'):
    line = line.strip()
    if len(line)<3:continue
    [did, senposVec, content, pred_label, pred_prob] = line.split('\t')
    dindex.append(int(did))
    test_text.append(content)
    predict_label.append(int(pred_label))
    pred_prob = pred_prob.replace('[','')
    pred_prob = pred_prob.replace(']','')
    x = [float(item) for item in pred_prob.split()]
    predict_prob.append(x)

def countWords(s):
    words = split(s)
    return len(words)

result = pd.DataFrame()
result['test_text'] = test_text
result['predict_label'] = predict_label
result['predict_prob'] = predict_prob    
result['predict_weight'] = predict_weight
result['predict_weight_prob'] = predict_weight_prob.tolist()
result['id'] = dindex

#label = {"利好":1, "中性":2, "利空":3}
trueid=[]
truelabel=[]
for line in open("true.txt") :
    line = line.strip()
    [did,label] = line.split('\t')
    if did !=None:
            if cmp(label,'利好') ==0: label = 1
            if cmp(label,'中性') ==0: label = 2
            if cmp(label,'利空') ==0: label = 3
    trueid.append(int(did))
    truelabel.append(label)
true = pd.DataFrame()
true['label'] = truelabel
true.index = trueid
true = true.drop(173687)
true = true.drop(177640)

#test set of microblog
testid=[]
testlabel=[]
testcontent=[]
for line in open("segged_test_sentpred.txt"):
    line = line.strip()
    [did, text, pred, pred_prob] = line.split('\t')
    testid.append(int(did))
    testlabel.append(int(pred))
    testcontent.append(text)
testwei = pd.DataFrame()
testwei['label']=testlabel
testwei.index=testid  
testwei = testwei.drop(173687)
testwei = testwei.drop(177640)    
result['word_count'] = [countWords(test_text[i]) for i in range(len(testcontent))] 

unsentence = np.mean(true['label'] == testwei['label'])

def cmpacc(cut_L, result, true, testwei):
    cut_idx = list(set(result[(result['word_count'] <= cut_L)].index))
    test_cut = result.drop(cut_idx)
    keep_wb_idx = list(set(test_cut['id']))   
    true_cut = true.ix[keep_wb_idx]
    true_cut = true_cut.sort_index()
    testwei_cut = testwei.ix[keep_wb_idx]
    testwei_cut = testwei_cut.sort_index()
    testwei_pred_sen = testwei.ix[list(set(testwei.index).difference(set(keep_wb_idx)))]
    
    '''
    cut_idx = list(set(result[(result['word_count'] <= cut_L)].index))
    test_cut = result.drop(cut_idx)
    true_cut = true.drop(cut_idx)
    '''
    
    predict_prob = test_cut['predict_prob']
    predict_prob.index = range(len(predict_prob))
    
    #不分句结果预测

    
    #软投票不加权
    soft_unweight = pd.DataFrame()
    soft_unweight['predict_1'] = [predict_prob[i][0] for i in range(len(predict_prob))]
    soft_unweight['predict_2'] = [predict_prob[i][1] for i in range(len(predict_prob))]
    soft_unweight['predict_3'] = [predict_prob[i][2] for i in range(len(predict_prob))]
    soft_unweight.index = test_cut['id']
    
    idx = []
    pred_soft_unweighted = []
    neg_prob = 0
    pos_prob = 0
    neu_prob = 0
    for i in true_cut.index:
        temp = soft_unweight[soft_unweight.index == i]
        idx.append(i)
        neg_prob = sum(temp['predict_3'])
        pos_prob = sum(temp['predict_1'])
        neu_prob = sum(temp['predict_2'])   
        if (max(neg_prob, pos_prob, neu_prob) == neg_prob):
            pred_soft_unweighted.append(3)
        if (max(neg_prob, pos_prob, neu_prob) == pos_prob):
            pred_soft_unweighted.append(1)
        if (max(neg_prob, pos_prob, neu_prob) == neu_prob):
            pred_soft_unweighted.append(2)
    
    pred_unweighted_res = pd.DataFrame()
    pred_unweighted_res['label'] = pred_soft_unweighted
    pred_unweighted_res['index'] = idx
    testwei_pred_sen['index'] = testwei_pred_sen.index
    res = pred_unweighted_res.append(testwei_pred_sen)
    res.index = res['index']
    res = res.sort_index()
    unweighted = np.mean(true['label'] == res['label'])
     
    predict_weight_prob = test_cut['predict_weight_prob']
    predict_weight_prob.index = range(len(predict_weight_prob))
    
    #软投票加权
    soft_weight = pd.DataFrame()
    soft_weight['predict_1'] = [predict_prob[i][0]*predict_weight_prob[i][1] for i in range(len(predict_prob))]
    soft_weight['predict_2'] = [predict_prob[i][1]*predict_weight_prob[i][1] for i in range(len(predict_prob))]
    soft_weight['predict_3'] = [predict_prob[i][2]*predict_weight_prob[i][1] for i in range(len(predict_prob))]
    soft_weight.index = test_cut['id']
    
    pred_soft_weighted = []
    neg_prob = 0
    pos_prob = 0
    neu_prob = 0
    #label = {"利好":1, "中性":2, "利空":3}
    for i in true_cut.index:
        temp = soft_weight[soft_weight.index == i]
        neg_prob = sum(temp['predict_3'])
        pos_prob = sum(temp['predict_1'])
        neu_prob = sum(temp['predict_2'])   
        if (max(neg_prob, pos_prob, neu_prob) == neg_prob):
            pred_soft_weighted.append(3)
        if (max(neg_prob, pos_prob, neu_prob) == pos_prob):
            pred_soft_weighted.append(1)
        if (max(neg_prob, pos_prob, neu_prob) == neu_prob):
            pred_soft_weighted.append(2)
    
    pred_weighted_res = pd.DataFrame()
    pred_weighted_res['label'] = pred_soft_weighted
    pred_weighted_res['index'] = idx
    res = pred_weighted_res.append(testwei_pred_sen)
    res.index = res['index']
    res = res.sort_index()
    weighted = np.mean(true['label'] == res['label'])
    
    return unweighted, weighted

#unsentence = []
unweighted = []
weighted = []
for L in range(11):
    [unwei, wei] = cmpacc(L, result, true,testwei)
    #unsentence.append(unsen)
    unweighted.append(unwei)
    weighted.append(wei)    

L = list(range(11))
plt.plot(L, unweighted, 'b')#,label=$cos(x^2)$)
plt.plot(L, weighted, 'r') 
#plt.plot(L,unsentence,'y')  

diff = [weighted[i] - unweighted[i] for i in range(len(weighted))] 
plt.plot(L, diff,'k')
