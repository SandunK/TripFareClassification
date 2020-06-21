import matplotlib.pyplot as plt
import datetime

from matplotlib.cm import rainbow
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd


# load data from csv
df_train = pd.read_csv('../input/fareclassification/train.csv') 
df_test = pd.read_csv('../input/fareclassification/test.csv')
df_sampleSubmit = pd.read_csv('../input/fareclassification/sample_submission.csv')

# Convert labels into 1,0
df_train = df_train.replace(to_replace = ['correct','incorrect'],value = ['1','0'])
df_train = df_train.astype({"label": int}) 

#Get all numerical and categorical coloms
numerical_cols = df_train.columns[df_train.dtypes != "object"].values
cate_cols = df_train.columns[df_train.dtypes == "object"].values

#impute null values using mean
simple_imputer = SimpleImputer(strategy='mean')
df_train[numerical_cols] = simple_imputer.fit_transform(df_train[numerical_cols])

# add new feature duration calculated using pickup and drop time
for column in cate_cols:
    parsed_time=pd.to_datetime(df_train[column])
    time=[datetime.datetime.time(d) for d in parsed_time]
    df_train[column]=[(t.hour * 60 + t.minute) * 60 + t.second for t in time ]

df_train['duration'] = df_train['drop_time']-df_train['pickup_time']

# add new feature "distance" for trip distance calculated using lat and long
def haversine(row):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = row['pick_lon']
    lat1 = row['pick_lat']
    lon2 = row['drop_lon']
    lat2 = row['drop_lat']

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

df_train['distance'] = df_train.apply(haversine, axis=1)
df_train = df_train.drop(['pick_lon','pick_lat','drop_lon','drop_lat','pickup_time','drop_time'],axis=1)

# plot feature co-relation matrix
from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 8
plt.matshow(df_train.corr())
plt.yticks(np.arange(df_train.shape[1]), df_train.columns)
plt.xticks(np.arange(df_train.shape[1]), df_train.columns)
plt.colorbar()

# Feature sampling
Y = df_train['label']
X = df_train.drop(['label','tripid'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.33,
    shuffle=True,
    stratify=Y,
    random_state=22
)


## parameter search for LGBM
clf = LGBMClassifier(n_estimators=800, learning_rate=0.05)
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}

n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                   n_iter=n_iter_search)

random_search.fit(X_train, y_train)

##Function to report search results
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

report(random_search.cv_results_)


## Ensembling
##lgb_model = LGBMClassifier(n_estimators=1500, learning_rate=0.05)
##xgb_model = XGBClassifier(n_estimators=1500, learning_rate=0.01)
##rf_model = RandomForestClassifier(bootstrap = True,max_depth=5,criterion = 'entropy',max_features=5,min_samples_leaf = 1,min_samples_split = 4,n_estimators = 1500)
##
##lgb_model.fit(X_train, y_train)
##xgb_model.fit(X_train, y_train)
##rf_model.fit(X_train, y_train)
##
##print("LightGBM Score     : ",lgb_model.score(X_test, y_test))
##print("XGBoost Score      : ",xgb_model.score(X_test, y_test))
##print("RandomForests Score: ",rf_model.score(X_test, y_test))

##Stacking Classifiers
##from sklearn.ensemble import StackingClassifier
##from sklearn.linear_model import LogisticRegression
##
##estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)]
##clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
##clf.fit(X_train, y_train)
##print("Ensemble Score: ", clf.score(X_test, y_test))
##clf.fit(X, Y)

# get aucscore

ypreds = clf.predict(X_test)
roc_auc_score(y_test, ypreds)

# predict for test data
df_test=df_test.drop(['tripid'],axis=1)
for column in cate_cols:
    parsed_time=pd.to_datetime(df_test[column])
    time=[datetime.datetime.time(d) for d in parsed_time]
    df_test[column]=[(t.hour * 60 + t.minute) * 60 + t.second for t in time ]

df_test['duration'] = df_test['drop_time']-df_test['pickup_time']
df_test['distance'] = df_test.apply(haversine, axis=1)
df_test = df_test.drop(['pick_lon','pick_lat','drop_lon','drop_lat','pickup_time','drop_time'],axis=1)

predictions = clf.predict(df_test)

# write predictions into output file
# Make sure we have the rows in the same order
np.testing.assert_array_equal(df_test.index.values, 
                              df_sampleSubmit.index.values)

# Save predictions to submission data frame
df_sampleSubmit["prediction"] = predictions[:]

df_sampleSubmit = df_sampleSubmit.astype({"prediction": int}) 
df_sampleSubmit.to_csv('my_submission.csv', index=False)
