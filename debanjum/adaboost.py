#!/usr/bin/env python

# Import Modules
import pandas as pd
import numpy as np
#import xgboost as xgb
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

# Load Training, Testing Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Extract Target Training Output into a Numpy Array
target = np.array(train['target'])

# Clean Test Data
testid = test['ID']
test = test.drop(['ID'], axis=1)

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
scaler = StandardScaler().fit(train)        # find mean and std based on training data to normalize for zero mean and unit variance
train = scaler.transform(train)             # normalize training data
test = scaler.transform(test)               # normalize testing data base on normalization fit for training data

# AdaBoost Classification
clf=AdaBoostClassifier(n_estimators=1800)
clf = clf.fit(train,target)
scores = cross_val_score(clf, train, target)
print "AdaBoost Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())
abpredict = clf.predict_proba(test)[:,1]
submit = pd.DataFrame({'ID': testid, 'PredictedProb': abpredict})
submit.to_csv("adaboost_submission.csv", index=False)
