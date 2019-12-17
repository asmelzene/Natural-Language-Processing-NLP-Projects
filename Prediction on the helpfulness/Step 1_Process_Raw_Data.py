'''
Only need to run once for once dataset

input: 
    reviews_Amazon_Instant_Video_5.json

output:
    #processed_data.pckl: processed data with 4 types of features
    #processed_data_withUSR.pckl: processed raw data 5 types of features 
    trainset.pckl
    testset.pckl
    userdict.pckl
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

def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization
    # https://github.com/Tsakunelson/Model_based_recommendation_engine/blob/master/2_Implementing_FunkSVD.ipynb
    INPUT:
    ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate 
    iters - (int) the number of iterations
    
    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    movie_mat - (numpy array) a latent feature by movie matrix
    '''
    n_users =len(ratings_mat) # number of rows in the matrix
    n_movies = ratings_mat[0,:].size # number of movies in the matrix
    num_ratings = ratings_mat.size  # total number of ratings in the matrix
    
    user_mat = np.random.rand(n_users, latent_features) # user matrix filled with random values of shape user x latent 
    movie_mat = np.random.rand(latent_features, n_movies)  # movie matrix filled with random values of shape latent x movies
    
    sse_accum = 0
    
    print("Optimization Statistics")
    print("Iterations | Mean Squared Error ")

    for epoch in range(iters):
        # update our sse
        old_sse = sse_accum
        sse_accum = 0
        
        # For each user-movie pair
        for i in range(n_users):
            for j in range(n_movies):
                # if the rating exists
                if ratings_mat[i,j] > 0:
                    
                    # compute the error as the actual minus the dot product of the user and movie latent features
                    error = ratings_mat[i,j]-np.dot(user_mat[i,:],movie_mat[:,j])
                    sse_accum += error**2
                    for k in range(latent_features):
                        
                        user_mat[i,k] +=  2*learning_rate*error*movie_mat[k,j]  # 无正则
                        movie_mat[k,j] +=  2*learning_rate*error*user_mat[i,k]

        if epoch % 50 == 0:      
        # print results for iteration
            print("%d \t\t %f" % (epoch, sse_accum / num_ratings))
    return user_mat, movie_mat



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
    

#    f = open('processed_data.pckl', 'wb')
#    pickle.dump(dataset, f)
#    f.close()
    
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
#    
#    f = open('processed_data_withUSR.pckl', 'wb')
#    pickle.dump(dataset, f)
#    f.close()
    
    maxtraintime = datetime.strptime('2014-01-01', '%Y-%m-%d')
    trainset = dataset[dataset["unixReviewTime"] <= maxtraintime]
    testset = dataset[dataset["unixReviewTime"] > maxtraintime]
    print("train size:{} test size:{} data size:{} test ratio: {:.3f}".format(len(trainset),len(testset),len(dataset), len(testset)/len(dataset)))
    
    f = open('trainset.pckl', 'wb')
    pickle.dump(trainset, f)
    f.close()
    f = open('testset.pckl', 'wb')
    pickle.dump(testset, f)
    f.close()
    
    trainset.reset_index(drop=True, inplace=True)
    useritem = trainset.groupby(['reviewerID','asin'], as_index=False)['percent_helpful'].mean()
    
    a = useritem.pivot(index='reviewerID', columns='asin', values='percent_helpful')
    nanratio=np.sum(np.sum(np.isnan(a.values).astype(int)))/(a.shape[0]*a.shape[1])
    print("(sparse matrix) nan ratio: {}".format(nanratio)) # 0.9977750950421623 sparse matrix
    

    helpmat = useritem.pivot(index='reviewerID', columns='asin', values='percent_helpful')
    
    
    user_mat, prod_mat = FunkSVD(helpmat.values, latent_features=5, learning_rate=0.0001, iters=100)
    
    userdict = {}
    for i in range(len(helpmat)):
        userdict.update( {helpmat.index[i]: user_mat[i]})           
        
    
    f = open('userdict.pckl', 'wb')
    pickle.dump(userdict, f)
    f.close()
    

data_preprocess()



