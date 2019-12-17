# -*- coding=utf-8 -*-
import os
import jieba
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

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def is_cn_char(qword):
    return ord(qword[0]) > 127


def get_cnstr(ori_str):
    cn_str = ''
    for c in ori_str:
        if is_cn_char(c):
            cn_str = cn_str + c
    return cn_str


def tokenize(sentence):
    cn_sent = get_cnstr(sentence)
    term_list = jieba.lcut(cn_sent, cut_all=False)
    final_term_list = [term for term in term_list if is_cn_char(term)]
    return final_term_list


def liststr(ourlist, sep_str=' '):
    ourstr = ''
    for item in ourlist: ourstr = ourstr + sep_str + item
    if len(ourstr) > 1:
        return ourstr[1:]
    else:
        return ourstr


def N_Gram(termList, N, sep='-'):
    ngramTermList = []
    for i in range(0, len(termList) - N + 1):
        i_ngramTerm = ''
        for j in range(0, N):
            current_term = termList[i + j]
            current_term = current_term.replace(sep, '')
            i_ngramTerm = i_ngramTerm + sep + current_term
        ngramTermList.append(i_ngramTerm[1:])
    return ngramTermList


def seg_for_docs_in_a_file(input_filename, input_sep, content_idx,  output_fileanme, t_len=0, did_idx=None, label_idx=None, out_sep='\t',
                           n_gram=1):
    print 'Word Segmentation...'
    segged_file = open(output_fileanme, 'w')
    for line in open(input_filename):
        line = line.strip('\n')
        if len(line) == 0: continue
        content = line.split(input_sep)[content_idx]
        if label_idx !=None:
            label = line.split(input_sep)[label_idx]
            if cmp(label,'利好') ==0: label='1'
            if cmp(label,'中性') ==0: label='2'
            if cmp(label,'利空') ==0: label='3'
            
        if did_idx != None:
            did = line.split(input_sep)[did_idx]
        seg_list = tokenize(content)
        term_list = [term for term in seg_list if len(term) > t_len]
        if n_gram > 1: term_list = N_Gram(seg_list, n_gram)
        if did_idx != None and label_idx !=None:
            segged_file.write(did + out_sep  + label + out_sep+ liststr(term_list) + '\n')
        if label_idx != None and did_idx ==None:
            segged_file.write(label + out_sep+  liststr(term_list) + '\n')
        if label_idx == None and did_idx !=None:
            segged_file.write(did + out_sep+  liststr(term_list) + '\n')
    segged_file.close()

seg_for_docs_in_a_file('train.txt', '\t', 1, 'segged_train.txt', did_idx=0, label_idx=-1)
seg_for_docs_in_a_file('test.txt', '\t', 1, 'segged_test.txt', did_idx=0)
