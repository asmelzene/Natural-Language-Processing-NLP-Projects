import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostRegressor


trainset = pickle.load(open('trainset.pckl', 'rb'))
testset = pickle.load(open('testset.pckl', 'rb'))
userdict = pickle.load(open('userdict.pckl', 'rb'))
print(trainset.shape[0])
print(trainset.shape[1])
print(testset.shape[0])
print(testset.shape[1])

trainset.reset_index(drop=True, inplace=True)
testset.reset_index(drop=True, inplace=True)

funkfeat = []
for i in range(len(trainset)):
    row = trainset.iloc[i,:]
    userid = row['reviewerID']
    if userid in userdict:
        funkfeat.append(userdict[userid])
    else:
        na = [float("nan")]*5
        funkfeat.append(na)
funkfeat = pd.DataFrame(funkfeat)
funkfeat.columns = ['USR_svd{}'.format(i) for i in range(funkfeat.shape[1])]

trainset = pd.concat([trainset, funkfeat], axis=1)

funkfeat = []
for i in range(len(testset)):
    row = testset.iloc[i,:]
    userid = row['reviewerID']
    if userid in userdict:
        funkfeat.append(userdict[userid])
    else:
        na = [float("nan")]*5
        funkfeat.append(na)
funkfeat = pd.DataFrame(funkfeat)
funkfeat.columns = ['USR_svd{}'.format(i) for i in range(funkfeat.shape[1])]

testset = pd.concat([testset, funkfeat], axis=1)

print(trainset.shape[0])
print(trainset.shape[1])
print(testset.shape[0])
print(testset.shape[1])



def create_ugr(trainset, testset):   

    train = trainset.copy()
    test = testset.copy()
    
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
    trainarray = pd.DataFrame(imp.fit_transform(train))
    testarray = pd.DataFrame(imp.transform(test))
    trainarray.columns = train.columns
    testarray.columns = test.columns

    return trainarray, train_targets, testarray, test_targets


def apply_models(regmodels, train_x, train_y, test_x, test_y):

    regmodels.fit(train_x, train_y)
    pred_train = np.array(regmodels.predict(train_x))
    pred_train = [x if x >= 0 else 0 for x in pred_train]

    pred = np.array(regmodels.predict(test_x))
    pred = [x if x >= 0 else 0 for x in pred]

    trainmse = mean_squared_error(train_y, pred_train)
    trainmae = mean_absolute_error(train_y, pred_train)
    
    testmse = mean_squared_error(test_y, pred)
    testmae = mean_absolute_error(test_y, pred)
   
    return trainmse, trainmae, testmse, testmae

def model_contrast(train_x, train_y, test_x, test_y, case="all"):

    res = {}

    print ('LinearRegression')
    res.update({"LinearRegression": apply_models(LinearRegression(), train_x, train_y, test_x, test_y)})

    print ('Decision Tree ')
    res.update({"DecisionTreeRegressor": apply_models(DecisionTreeRegressor(max_depth=90), train_x, train_y, test_x, test_y)})

    print ('Gradient Boosting')
    res.update({"GradientBoosting": apply_models(GradientBoostingRegressor(), train_x, train_y, test_x, test_y)})


    print ('LightGBM')
    gbm = lgb.LGBMRegressor(boosting_type='gbdt',
                          objective = 'regression',
                          metric = 'mse',
                          verbose_eval=500,
                          num_boost_round=100,
                          random_state = 2019
                          )
    res.update({"LightGBM": apply_models(gbm, train_x, train_y, test_x, test_y)})

    print ('Adaboost')
    res.update({"Adaboost": apply_models(AdaBoostRegressor(n_estimators=100), train_x, train_y, test_x, test_y)})


    print ('XGBoost')
    xgbmodel = xgb.XGBRegressor(
                      verbosity=0,
                      n_estimators=100,
                      random_state = 2019
                      )   
    res.update({"XGBoost": apply_models(xgbmodel, train_x, train_y, test_x, test_y)})

    print ('Random Forest')
    start = time.time()
    res.update({"RandomForest": apply_models(RandomForestRegressor(), train_x, train_y, test_x, test_y)})
    print('It took {0:0.2f} seconds for RF training'.format(time.time() - start)) 

    print ('SVM')
    start = time.time()
    res.update({"SVM": apply_models(svm.SVR(kernel='rbf'), train_x, train_y, test_x, test_y)})  
    print('It took {0:0.2f} seconds for SVM training'.format(time.time() - start)) 

    print(res)

    f = open('res_{}.pckl'.format(case), 'wb')
    pickle.dump(res, f)
    f.close()

train_x, train_y, test_x, test_y = create_ugr(trainset, testset)  

#######  STR  +  UGR  + GALC + SEN + USR
print("----------------------- ALL Features -----------------------")
print(train_x.shape[1])
model_contrast(train_x, train_y, test_x, test_y, case="all")


#######  STR  +  UGR  + GALC + SEN 
print("----------------------- ALL Features without USR -----------------------")

keepcol = [x for x in train_x.columns if not x.startswith("USR")]
trainpart = train_x[keepcol]
testpart = test_x[keepcol]
print(trainpart.shape[1])
model_contrast(trainpart, train_y, testpart, test_y, case="allbutUSR")


########  
print("----------------------- STR  +  UGR  Features -----------------------")
keepcol = [x for x in train_x.columns if x.startswith("STR") or x.startswith("UGR")]
trainpart = train_x[keepcol]
testpart = test_x[keepcol]
print(trainpart.shape[1])
model_contrast(trainpart, train_y, testpart, test_y, case="STR+UGR")


########  
print("----------------------- UGR  Features -----------------------")
keepcol = [x for x in train_x.columns if x.startswith("UGR")]
trainpart = train_x[keepcol]
testpart = test_x[keepcol]
print(trainpart.shape[1])
model_contrast(trainpart, train_y, testpart, test_y, case="UGR")

#######  
print("----------------------- STR  Features -----------------------")
keepcol = [x for x in train_x.columns if x.startswith("STR")]
trainpart = train_x[keepcol]
testpart = test_x[keepcol]
print(trainpart.shape[1])
model_contrast(trainpart, train_y, testpart, test_y, case="STR")


    
'''
#show results
import pickle
import numpy as np
f = open('/content/gdrive/My Drive/Colab Notebooks/res_all.pckl', 'rb')
res_all = pickle.load(f)
f.close()


f = open('/content/gdrive/My Drive/Colab Notebooks/res_allbutUSR.pckl', 'rb')
res_allbutUSR = pickle.load(f)
f.close()


f = open('/content/gdrive/My Drive/Colab Notebooks/res_STR+URG.pckl', 'rb')
res_STRURG = pickle.load(f)
f.close()

f = open('/content/gdrive/My Drive/Colab Notebooks/res_UGR.pckl', 'rb')
res_URG = pickle.load(f)
f.close()

f = open('/content/gdrive/My Drive/Colab Notebooks/res_STR.pckl', 'rb')
res_STR = pickle.load(f)
f.close()


res = {}
res["res_all"] = res_all
res["res_allbutUSR"] = res_allbutUSR
res["res_STR+URG"] = res_STRURG
res["res_URG"] = res_URG
res["res_STR"] = res_STR

for model in res:
    print(model)
    for reg in res[model]:
        print(reg)

        print(res[model][reg])
    print()
            
'''





