# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 19:05:06 2017

@author: zhao
"""
from sklearn import svm
import numpy as np
import pandas as pd
from string import *
import matplotlib.pyplot as plt

train_data = []
train_lab = []
test_data = []
for line in open('train_senpos_sentpred_binary.txt'):
    line = line.strip()
    if len(line)<3:continue
    [did,senposVec, true_label, content, pred_label, pred_prob] = line.split('\t')
    senposVec = senposVec.replace('[','')
    senposVec = senposVec.replace(']','')
    x = [float(item) for item in senposVec.split(', ')]
    #原本是若句子所在微博的真实label与句子被预测的label相同，说明该条句子有意义 标为1,而现在用置信度取代0/1
    pred_prob = pred_prob.replace('[', '')
    pred_prob = pred_prob.replace(']', '')
    tmp = [float(item) for item in pred_prob.split()]
    y = tmp[0]
    train_data.append(x)
    train_lab.append(y)

for line in open('test_senpos_sentpred_binary.txt'):
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

#model 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(random_state=0, n_estimators=200, oob_score = True) 
#from sklearn.svm import SVR
#clf = SVR(kernel = "linear") 
clf.fit(train_X, train_y)
#clf.score(train_X, train_y)

predict_weight_reg = clf.predict(test_X)
#predict_weight_prob = clf.predict_proba(test_X)
dindex = []
predict_label = []
predict_prob = []
test_text = []
for line in open('test_senpos_sentpred_binary.txt'):
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
result['predict_weight_reg'] = predict_weight_reg
result.index = dindex

#label = {"利好":1, "利空":3}
trueid=[]
truelabel=[]
for line in open("true_binary.txt") :
    line = line.strip()
    [did,label] = line.split('\t')
    if did !=None:
            if cmp(label,'利好') ==0: label = 1
            #if cmp(label,'中性') ==0: label = 2
            if cmp(label,'利空') ==0: label = 3
    trueid.append(int(did))
    truelabel.append(label)
true = pd.DataFrame()
true['label'] = truelabel
true.index = trueid
#true = true.drop(173687)
#true = true.drop(177640) 这两列就是中性文本

#test set of microblog
testid = []
testtext = []
testlabel = []
for line in open("segged_test_sentpred_binary.txt"):
    line = line.strip()
    [did, text, pred, pred_prob] = line.split('\t')
    testid.append(int(did))
    testtext.append(text)
    testlabel.append(int(pred))
testwei = pd.DataFrame()
testwei['label'] = testlabel
testwei['text'] = testtext   
testwei['word_count'] = [countWords(testwei['text'].ix[i]) for i in range(len(testwei))]
testwei.index = testid  
#testwei = testwei.drop(173687)
#testwei = testwei.drop(177640)

def cmp_cut_acc(cut_L, result, true):
    #cut_L=5
    cut_idx = list(set(testwei[(testwei['word_count'] <= cut_L)].index))
    test_cut = result.drop(cut_idx)
    true_cut = true.drop(cut_idx)
    testwei_cut = testwei.drop(cut_idx)
    
    #不分句结果预测
    unsentence = np.mean(true_cut['label'] == testwei_cut['label'])    

    predict_prob = test_cut['predict_prob']
    predict_prob.index = range(len(predict_prob))
    
    #软投票不加权
    soft_unweight = pd.DataFrame()
    soft_unweight['predict_1'] = [predict_prob[i][0] for i in range(len(predict_prob))]
    soft_unweight['predict_3'] = [predict_prob[i][1] for i in range(len(predict_prob))]
    soft_unweight.index = test_cut.index
    
    pred_soft_unweighted = []
    neg_prob = 0
    pos_prob = 0
    for i in true_cut.index:
        temp = soft_unweight[soft_unweight.index == i]
        neg_prob = sum(temp['predict_3'])
        pos_prob = sum(temp['predict_1'])
        if (max(neg_prob, pos_prob) == neg_prob):
            pred_soft_unweighted.append(3)
            continue
        if (max(neg_prob, pos_prob) == pos_prob):
            pred_soft_unweighted.append(1)
            continue
    
    unweighted = np.mean(true_cut['label'] == pred_soft_unweighted)
    
    predict_weight_reg = test_cut['predict_weight_reg']
    predict_weight_reg.index = range(len(predict_weight_reg))
    
    #软投票加权
    soft_weight = pd.DataFrame()
    soft_weight['predict_1'] = [predict_prob[i][0]*predict_weight_reg[i] for i in range(len(predict_prob))]
    soft_weight['predict_3'] = [predict_prob[i][1]*predict_weight_reg[i] for i in range(len(predict_prob))]
    soft_weight.index = test_cut.index
    
    pred_soft_weighted = []
    neg_prob = 0
    pos_prob = 0
    #label = {"利好":1, "利空":3}
    for i in true_cut.index:
        temp = soft_weight[soft_weight.index == i]
        neg_prob = sum(temp['predict_3'])
        pos_prob = sum(temp['predict_1']) 
        if (max(neg_prob, pos_prob) == neg_prob):
            pred_soft_weighted.append(3)
            continue
        if (max(neg_prob, pos_prob) == pos_prob):
            pred_soft_weighted.append(1)
            continue
    
    weighted = np.mean(true_cut['label'] == pred_soft_weighted)
    
    return unsentence,unweighted, weighted

#cmp_cut_without_combination
unsentence = []
unweighted = []
weighted = []
for L in range(51):
    [unsen, unwei, wei] = cmp_cut_acc(L, result, true)
    unsentence.append(unsen)
    unweighted.append(unwei)
    weighted.append(wei)    
print '\a'*7
L = list(range(51))
plt.plot(L, unweighted, 'b', label = 'unweighted')
plt.plot(L, weighted, 'r--', label = 'weighted')    
plt.plot(L,unsentence, 'g', label = 'unseg') 
plt.legend()

diff = [weighted[i] - unweighted[i] for i in range(len(weighted))] 
plt.plot(L, diff,'k',label='diff between weight and unweight')
plt.legend()

diff = [weighted[i] - unsentence[i] for i in range(len(weighted))] 
plt.plot(L, diff,'k',label='diff between weight and unseg')
plt.legend()
