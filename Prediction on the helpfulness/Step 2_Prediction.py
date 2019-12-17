'''
Prediction Task

input: processed_data_withUSR.pckl

output:
results of different models 
'''

import pickle
import numpy as np
from textblob import TextBlob
from textblob import Word
import pandas as pd
import pickle
import time
import re
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#####   Classify
from sklearn.preprocessing import Imputer
   
     
def apply_cross_validate(regmodels, data, cv=5):
    res = {}
    mse = []
    mae = []
    medianae = []
    for i in range(cv):
        train, test = train_test_split(data, train_size=0.8)
       
        traintext = [' '.join(x) for x in train["words"]]
        testtext = [' '.join(x) for x in test["words"]]
        
        # create UGR features -> cannot do it in advance since it is dependent in training data

        tfidf = TfidfVectorizer(max_df=0.8, min_df=3, max_features=1000).fit(traintext) 
        train_tfidf = tfidf.transform(traintext).todense()   # 转化为更直观的一般矩阵
        vocabulary = tfidf.vocabulary_
        sorted_vocabulary = sorted(vocabulary.items(), key=lambda item: item[1], reverse=False)
        
        test_tfidf = tfidf.transform(testtext).todense()   # 转化为更直观的一般矩阵
        
        train_tfidf = pd.DataFrame(train_tfidf)
        train_tfidf.columns = ["UGR_"+x[0] for x in sorted_vocabulary]
        
        test_tfidf = pd.DataFrame(test_tfidf)
        test_tfidf.columns = ["UGR_"+x[0] for x in sorted_vocabulary]
        
        train_targets = train['percent_helpful'].values
        test_targets = test['percent_helpful'].values
        
        del train['percent_helpful']
        del test['percent_helpful']
        
        del train['reviewerID'], train['asin'], train['reviewText'], train['words'], train['unixReviewTime'], train['review_helpful'], train['overall_votes'], train['helpful_votes']
        del test['reviewerID'], test['asin'], test['reviewText'], test['words'], test['unixReviewTime'], test['review_helpful'], test['overall_votes'], test['helpful_votes']
        
        train = pd.concat([train.reset_index(drop=True), train_tfidf.reset_index(drop=True)], axis=1)
        test = pd.concat([test.reset_index(drop=True), test_tfidf.reset_index(drop=True)], axis=1)
        
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        train = imp.fit_transform(train)
        test = imp.transform(test)

        regmodels.fit(train, train_targets)
        pred = np.array(regmodels.predict(test))
        pred = [x if x >= 0 else 0 for x in pred]
        
        mse.append(mean_squared_error(test_targets,pred))
        mae.append(mean_absolute_error(test_targets,pred))
        medianae.append(median_absolute_error(test_targets,pred))
            
            
    res.update({'mean_squared_error': mse})
    res.update({'mean_absolute_error': mae})
    res.update({'median_absolute_error': medianae})
    return res, regmodels

#res=apply_cross_validate(LinearRegression(), dataset, cv=5)

def model_contrast(dataset, case):
    res = {}

    print ('LinearRegression')
    res.update({"LinearRegression": apply_cross_validate(LinearRegression(), data=dataset, cv=5)})

    print ('Decision Tree ')
    res.update({"DecisionTreeRegressor": apply_cross_validate(DecisionTreeRegressor(max_depth=90), data=dataset, cv=5)})

    print ('Gradient Boosting')
    res.update({"GradientBoosting": apply_cross_validate(GradientBoostingRegressor(), data=dataset, cv=5)})

    print ('Random Forest')
    start = time.time()
    res.update({"RandomForest": apply_cross_validate(RandomForestRegressor(), data=dataset, cv=5)})
    print('It took {0:0.2f} seconds for RF training'.format(time.time() - start)) 

    print ('SVM')
    start = time.time()
    res.update({"SVM": apply_cross_validate(svm.SVR(kernel='rbf'), data=dataset, cv=5)})  
    print('It took {0:0.2f} seconds for SVM training'.format(time.time() - start)) 
    
#    print(res)
    
    f = open('res_{}.pckl'.format(case), 'wb')
    pickle.dump(res, f)
    f.close()
    
    print (res)

###############

f = open('dataset_withUSR.pckl', 'rb')
dataset = pickle.load(f)
f.close()

dataset = dataset[dataset["overall_votes"]>0].reset_index(drop=True)



#######  STR  +  UGR  + GALC + SEN + USR
print("----------------------- ALL Features -----------------------")

model_contrast(dataset, case="all")


#######  STR  +  UGR  + GALC + SEN 
print("----------------------- ALL Features without USR -----------------------")

model_contrast(dataset, case="allbutUSR")


########  
print("----------------------- STR  +  UGR  Features -----------------------")

model_contrast(dataset, case="STR+URG")

########  
print("----------------------- UGR  Features -----------------------")

model_contrast(dataset, case="URG")
#######  
print("----------------------- STR  Features -----------------------")

model_contrast(dataset, case="STR")


res, tree = apply_cross_validate(GradientBoostingRegressor(), data=dataset, cv=5)


#    
#feature_importance = pd.DataFrame({"features":train.columns, "importance":tree.feature_importances_})
#feature_importance = feature_importance.sort_values("importance", ascending=False)
#a = feature_importance.iloc[:15, :]

#sns.barplot(a["features"], a["importance"], palette="Blues_d")
#plt.tick_params(axis="x", rotation=90) 
#plt.savefig("fi.png")
#

    
    
    
    

