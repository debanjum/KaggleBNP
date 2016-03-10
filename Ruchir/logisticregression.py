import loaddata
import writesubmission
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import time
from sklearn import linear_model
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Using all default values for now
classifier = LogisticRegression(penalty='l2')
imputer = Imputer()

start = time.time()
print 'Loading data...'
train_data = loaddata.train_data()
test_data = loaddata.test_data()
print '...done loading data. Took {0} seconds'.format(time.time() - start)

print 'Imputing missing X values in training data...'
start = time.time()
train_data['x'] = imputer.fit_transform(train_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

start = time.time()
print 'Training model...'
classifier.fit(train_data['x'], train_data['y'].tolist())
print '...model trained. Took {0} seconds'.format(time.time() - start)

print 'Imputing missing X values in testing data...'
start = time.time()
test_data['x'] = imputer.fit_transform(test_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

start = time.time()
print 'Generating predictions...'
y_hat = classifier.predict_proba(test_data['x'])
print '...predictions generated. Took {0} seconds'.format(time.time() - start)

start = time.time()
print 'Writing output...'
writesubmission.writesubmission(test_data['ids'], y_hat, 'foo.csv')
print '...output written. Took {0} seconds'.format(time.time() - start)

#print(metrics.classification_report(train_data['y'], y_hat))
#print(metrics.confusion_matrix(train_data['y'], y_hat))
#print pd.crosstab(train_data['y'], y_hat, rownames=['True'], colnames=['Predicted'], margins=True)