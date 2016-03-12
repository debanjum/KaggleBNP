#!/usr/bin/env python

# Import Modules
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

# Load Training, Testing Data
train = pd.read_csv('../data/data/train.csv')
test = pd.read_csv("../data/data/test.csv")

# Extract Target Training Output into a Numpy Array
target = np.array(train['target'])

# Clean Test Data
testid = test['ID']
test.drop(['ID'], axis=1, inplace=True)

# Extract Categorical Training Data, Transform into Float and Load into Training Data
#train2 = train.join([pd.get_dummies(train[column],prefix=column) for column in train.columns[2:8] if train[column].dtype == 'object']) #works when one category added for > memory error
#for column in data.columns[2:]:
#    if data[column].dtype == 'object':
#        pd.concat([train, pd.get_dummies(data[column])], axis=1)

# Remove Categorical Features
train = train[[column for column in train.columns[2:] if train[column].dtype != 'object']]
test = test[[column for column in test.columns if test[column].dtype != 'object']]

# Replace NaN values by feature medians in dataset
#  In training data
impute = Imputer(missing_values='NaN', strategy='median', axis=0)
impute.fit(train)
train =impute.transform(train)

#  In testing data
impute.fit(test)
test = impute.transform(test)

# Normalize Training and Testing Dataset
scaler = preprocessing.StandardScaler().fit(train)    # find mean and std based on training data to normalize for zero mean and unit variance
train = scaler.transform(train)                       # normalize training data
test = scaler.transform(test)                         # normalize testing data base on normalization fit for training data

# Decision Tree Classification
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)                   # create classifier
scores = cross_val_score(clf, train, target)                                                       # get cross validation scores
print "Decision Tree Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())   # print cross validation mean, deviation
dtpredict = clf.predict_proba(test)[:,1]                                                           # make predictions
submit = pd.DataFrame({'ID': testid, 'PredictedProb': dtpredict})
submit.to_csv("decisiontree_submission.csv", index=False)

# Random Forest Classification
clf=RandomForestClassifier(n_estimators=1800)
clf=clf.fit(cleaned,target)
scores = cross_val_score(clf, train, target)
print "Random Forest Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())
rfpredict = clf.predict_proba(test)[:,1]
submit = pd.DataFrame({'ID': testid, 'PredictedProb': rfpredict})
submit.to_csv("randomforest_submission.csv", index=False)

# AdaBoost Classification
clf=AdaBoostClassifier(n_estimators=1800)
scores = cross_val_score(clf, train, target)
print "AdaBoost Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())
abpredict = clf.predict_proba(test)[:,1]
submit = pd.DataFrame({'ID': testid, 'PredictedProb': abpredict})
submit.to_csv("adaboost_submission.csv", index=False)

# XGBoost Classification
#xgtrain = xgb.DMatrix(train, target); xgtest = xgb.DMatrix(test); 
xgboost_params = {"objective": "binary:logistic", "booster": "gbtree", "eval_metric": "auc", "eta": 0.01, "subsample": 0.75, "colsample_bytree": 0.68, "max_depth": 7}
#xgboost_params = {
#    'learning_rate': [0.01, 0.1],
#    'n_estimators': [100, 300, 1800],
#    'max_depth': [2, 7]
#}
clf = xgb.train(xgboost_params,xgtrain,num_boost_round=1800,verbose_eval=True,maximize=False) # train with xgboost
#xgboost = xgb.XGBClassifier()
#clf = GridSearchCV(xgboost, xgboost_params, n_jobs=-1, cv=StratifiedKFold(target, n_folds=5),scoring='roc_auc')
#scores = cross_val_score(clf, train, target)                                                 # get cross validation scores
#print "XGBoost Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())   # print cross validation mean, deviation
xgpredict = clf.predict(xgtest, ntree_limit=clf.best_iteration)                              # make predictions

# Write Prediction to File in Competition Submission Format 
submit = pd.DataFrame({'ID': testid, 'PredictedProb': xgpredict})
submit.to_csv("xgboost_submission.csv", index=False)

# Ensembeled Predict
submit = pd.DataFrame({'ID': testid, 'PredictedProb': (xgpredict+abpredict+rfpredict)/3})
submit.to_csv("xg_rf_ab_submission.csv", index=False)


#gbm = xgb.XGBClassifier()
#xgboost_params = {
#    'learning_rate': [0.01, 0.05, 0.1, 0.5],
#    'n_estimators': [100, 300, 1000],
#    'max_depth': [3, 7, 10, None],
#    "objective": "binary:logistic", 
#    "booster": "gbtree",
#    "eta": 0.01, 
#    "subsample": 0.75, 
#    "colsample_bytree": 0.68
#}
#gbm_grid = GridSearchCV(gbm, gbm_params, n_jobs=-1, cv=StratifiedKFold(target, n_folds=5),scoring='roc_auc')
#scores = cross_val_score(gbm_grid , train, target)
#
#xgpredict = cv.best_estimator_.predict(test)
#
## Write Prediction to File in Competition Submission Format 
#submit = pd.DataFrame({'ID': testid, 'PredictedProb': xgpredict})
#submit.to_csv("xgboost_submission.csv", index=False)
