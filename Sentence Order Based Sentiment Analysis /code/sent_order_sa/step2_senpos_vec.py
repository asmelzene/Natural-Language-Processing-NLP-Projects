# -*- coding: utf-8 -*-
import jieba
import re
def get_valid_dids(filename_list, did_index):
    dids =set()
    for filename in filename_list:
        for line in open(filename):
            if len(line)<2:continue
            did = line.split('\t')[did_index]
            dids.add(did)
    return dids


#改成中文标点
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
def changepoint(sen):
    while 1:
        mm = re.search(ur"[\u4e00-\u9fa5](\.|:|;|；|：)[\u4e00-\u9fa5]",sen.decode('utf8'))
        if mm:
            mm = mm.group()
            mm = mm.encode('utf-8')
            sen = sen.replace(mm, mm.replace(".", "。"))
            sen = sen.replace(mm, mm.replace(":", "。"))
            sen = sen.replace(mm, mm.replace("：", "。"))
            sen = sen.replace(mm, mm.replace("；", "。"))
            sen = sen.replace(mm, mm.replace(";", "。"))
        else:
            break
    return sen

#分句
def cut_sentence_new(words):
    words = (words).decode('utf8')
    start = 0
    i = 0
    token = ''
    sents = []
    punt_list = ']】#!?~。！？～'.decode('utf8')
    for word in words:
        if word in punt_list and token not in punt_list: #检查标点符号下一个字符是否还是标点
            sents.append(words[start:i+1])
            start = i+1
            i += 1
        else:
            i += 1
            token = list(words[start:i+2]).pop() # 取下一个字符
    if start < len(words):
        sents.append(words[start:])
    return sents

valid_dids = get_valid_dids(['segged_test.txt', 'segged_train.txt'],0)


result_file = open('train_senpos.txt','w')
for line in open('train.txt'):
    line = line.strip()
    if len(line)<3:continue
    [did, content, label] = line.split('\t')
    if not did in valid_dids:continue
    if cmp(label,'利好') ==0: label='1'
    if cmp(label,'中性') ==0: label='2'
    if cmp(label,'利空') ==0: label='3'
    o_cutted_sen =  cut_sentence_new(content)
    cutted_sen = [csen for csen in o_cutted_sen if len(csen)>2]
    for i in range(0, len(cutted_sen)):
        csen = cutted_sen[i]
        spvec = [0 for j in range(0,10)]
        if i <5: spvec[i]=1
        if not len(cutted_sen)-i>5: spvec[-(len(cutted_sen)-i)]=1
        seg_list = tokenize(csen)
        term_list = [term for term in seg_list if len(term) > 0]
        if len(term_list)<2:continue
        result_file.write(did + '\t' + str(spvec) + '\t' + label + '\t' + liststr(term_list) + '\n')
 
result_file = open('test_senpos.txt','w')
for line in open('test.txt'):
    line = line.strip()
    if len(line)<3:continue
    [did, content] = line.split('\t')
    if not did in valid_dids:continue
    o_cutted_sen =  cut_sentence_new(content)
    cutted_sen = [csen for csen in o_cutted_sen if len(csen)>2]
    for i in range(0, len(cutted_sen)):
        csen = cutted_sen[i]
        spvec = [0 for j in range(0,10)]
        if i <5: spvec[i]=1
        if not len(cutted_sen)-i>5: spvec[-(len(cutted_sen)-i)]=1
        seg_list = tokenize(csen)
        term_list = [term for term in seg_list if len(term) > 0]
        if len(term_list)<2:continue
        result_file.write(did + '\t' + str(spvec)  + '\t' + liststr(term_list) + '\n')    
