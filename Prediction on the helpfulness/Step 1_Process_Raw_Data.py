'''
Only need to run once for once dataset

input: reviews_Amazon_Instant_Video_5.json

output:
processed_data.pckl: processed data with 4 types of features
processed_data_withUSR.pckl: processed raw data 5 types of features 
'''


import pickle
import numpy as np
from textblob import TextBlob
from textblob import Word
import pandas as pd
import time
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime


def parse(path):
    g = open(path, 'r')
    for l in g:
        yield eval(l)

def data_preprocess():
    
    reviews_data = 'reviews_Amazon_Instant_Video_5.json' #数据量少
    reviews = ([])
    
    obj_reviews = parse(reviews_data)
    for i in obj_reviews:
        reviews.append(i)
    
    dataset = pd.DataFrame(reviews)
        
    #['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime'],
    dataset = dataset[['reviewerID', 'asin', 'helpful', 'reviewText', 'overall', 'unixReviewTime']]
    
    # target variable process   4
    dataset['helpful_votes'] = dataset.helpful.apply(lambda x: x[0])
    dataset['overall_votes'] = dataset.helpful.apply(lambda x: x[1])
    dataset['percent_helpful'] = round((dataset['helpful_votes'] / dataset['overall_votes']),2)
    dataset['percent_helpful'] = dataset['percent_helpful'].fillna(0)
    dataset['review_helpful'] = np.where((dataset.percent_helpful > 0.6) & (dataset.overall_votes > 5), 1, 0)
    
    
    del dataset['helpful']
    
    # create STR features 5
    
    dataset['STR_no_of_words'] = dataset['reviewText'].apply(lambda x: len(TextBlob(x).words))
    dataset['STR_no_of_sentences'] = dataset['reviewText'].apply(lambda x: len(TextBlob(x).sentences))
    dataset['STR_words_per_sentence'] = dataset.STR_no_of_words / dataset.STR_no_of_sentences
    dataset['STR_no_of_exclamation'] = dataset['reviewText'].apply(lambda x: len(re.findall('!', x))) 
    dataset['STR_no_of_question'] =  dataset['reviewText'].apply(lambda x: len(re.findall('\?', x)))
    
    
    # create GALC features
    with open("GALC.pkl", "rb") as fp:  
        galc = pickle.load(fp)     
    
    with open("stopwords_en.pkl", "rb") as fp:  
        stopwords = pickle.load(fp)    
    
    
    # remove stop words + lemmatize 
    words_list = []
    for i in tqdm(range(dataset.shape[0])):
        rtext = dataset["reviewText"].iloc[i]
        words = TextBlob(rtext).words      #  tokenize
        words = [str(Word(x).lemmatize()) for x in words]# lemmatize
        words = [x.lower() for x in words] 
        words = [x for x in words if x not in stopwords] # remove stop words
        words_list.append(words)
    dataset["words"] = words_list
    
    
    # calculate GALC features 38
    start = time.time()
    features_data = pd.DataFrame()
    for i in tqdm(range(dataset.shape[0])):
        features = {}
        for k in set(galc.values()):
            features.update({k:0})
        words = dataset["words"].iloc[i]
        for word in words:
            for key in galc:
                if word.startswith(key):
                    features[galc[key]] += 1 
        features_data = pd.concat([features_data, pd.DataFrame(features, index=[0])])
    
    print('It took {0:0.2f} seconds for GALC feature extraction '.format(time.time() - start)) # 37126 samples -> 5.30 minites / 335.72s
    
    features_data.reset_index(drop=True, inplace=True)
    features_data.columns = ["GALC_"+x for x in features_data.columns]
    dataset = pd.concat([dataset,features_data], axis=1)
    
    # create SEN features 
    dataset['SEN_subjectivity'] = dataset['reviewText'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    dataset['SEN_polarity'] = dataset['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
    

    f = open('processed_data.pckl', 'wb')
    pickle.dump(dataset, f)
    f.close()
    
    return dataset

dataset = data_preprocess()

'''
####
dataset = dataset[dataset["overall_votes"]>0].reset_index(drop=True)
dataset['unixReviewTime'] = dataset['unixReviewTime'].apply(lambda x: datetime.fromtimestamp(x))

histoverall = []
histSTR_no_of_words = []
histSTR_no_of_sentences = []
histSEN_polarity = []
dataset.index = dataset["reviewerID"]
for i in tqdm(range(len(dataset))):
    row = dataset.iloc[i,:]
    
    userinfo = dataset.loc[[row["reviewerID"]]]
    userinfo = userinfo[userinfo['unixReviewTime'] < row['unixReviewTime']] #取出以前的信息
    if len(userinfo) == 0:
        histoverall.append(float("nan"))
        histSTR_no_of_words.append(float("nan"))
        histSTR_no_of_sentences.append(float("nan"))
        histSEN_polarity.append(float("nan"))
    else:
        histoverall.append(np.mean(userinfo["overall"]))
        histSTR_no_of_words.append(np.mean(userinfo["STR_no_of_words"]))
        histSTR_no_of_sentences.append(np.mean(userinfo["STR_no_of_sentences"]))
        histSEN_polarity.append(np.mean(userinfo["SEN_polarity"]))

dataset["USR_overall"] = histoverall    
dataset["USR_STR_no_of_words"] = histSTR_no_of_words    
dataset["USR_STR_no_of_sentences"] = histSTR_no_of_sentences    
dataset["USR_SEN_polarity"] = histSEN_polarity    

f = open('processed_data_withUSR.pckl', 'wb')
pickle.dump(dataset, f)
f.close()
'''
