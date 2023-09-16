# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##Import relevant libraries
import numpy as np
import pandas as pd
from flaml import AutoML
import catboost
from IPython.display import display ##Need this to use display() on supercomp
from sklearn.metrics import mean_squared_error
import pickle
import random
from matplotlib import pyplot as plt

##Set random seed
random.seed(10)

##Load in full dataset (31 ensemble members)
df_full = pd.read_parquet('Final full dataset.parquet', engine = 'pyarrow')

print("Full dataset info:")
display(df_full.info())

##Split into train-test sets
##Need to be able to grab 'time' to define train and test sets
##Also will be using 'lat' and 'lon' as predictor variables so reset whole index
df_full.reset_index(inplace=True)

##Define start and end date of train set
train_start = pd.to_datetime('2006-01-02')
train_end = pd.to_datetime('2070-12-31')

##Define train set as all dates between daterange 2006-2070
df_train = df_full.loc[(df_full['time']>=train_start) & (df_full['time']<=train_end)]
##Define test set as all dates between daterange 2071-2080
df_test = df_full.loc[df_full['time']>train_end]

print('Train set preview:')
display(df_train)
print('Test set preview:')
display(df_test)

##Define X_train, y_train & X_test, y_test
X_train = df_train.drop(columns=['time', 'Ensemble_num'])
X_test = df_test.drop(columns=['time', 'Ensemble_num'])
y_train = X_train.pop("TREFMXAV_U")
y_test = X_test.pop("TREFMXAV_U")

print("X_train:")
display(X_train.info())
print("y_train:")
display(y_train.info())

##AutoML
##Create object
automl = AutoML()
##Fit to training data, testing one model type at a time
automl_settings = {
    "time_budget": 3600,  # in seconds
    "metric": 'mse',
    "task": 'regression',
    "estimator_list": ['catboost'],
    "early_stop": True,
    "eval_method": "cv",
    "n_splits": 10,
    "seed": 10  #42, 10, 25
}

##Train model
automl.fit(X_train=X_train, y_train=y_train, verbose=0, **automl_settings)

##Saving our trained model
print("Saving best Simple Regression Catboost model as 'Reg_catboost.pickle' file")
pickle.dump(automl, open("Reg_catboost.pickle", "wb"))

##Best estimator
print('Best Catboost learner:', automl.best_estimator)
print('Best hyperparmeter config:', automl.best_config)
print('Best MSE score on validation data: {0:.5g}'.format(automl.best_loss))    ##this is for when you use mse
print('Training duration of best run: {0:.5g} s'.format(automl.best_config_train_time))

##Make predictions
y_pred = automl.predict(X_test)
##Score model
print("MSE of y_test and y_pred:", mean_squared_error(y_test,y_pred))

##We want to save X_test (2006-2070) and y_pred (2071-2080) so that we can concatenate them together for visualisation
##and for analysing ensemble member/spatial/temporal differences in model performance
##Note that the automatic row index should be retained automatically in its first column (which we can set as index later)
##Saving just the important parts of df_test: time, location, ensemble number
observe_test = df_test.iloc[:,:5]
##Adding y_pred as new column
observe_test['y_pred'] = y_pred

display(observe_test)
print("Saving test & y_pred column to 'y_pred_reg_catboost.csv'")
observe_test.to_csv('y_pred_reg_catboost.csv')

##Feature Importance
##Defining list of features
features = ['lat', 'lon', 'Day_of_year', 'Year', 'FLNS', 'FSNS', 'PRECT', 'PRSN', 'QBOT', 'TREFHT', 'UBOT', 'VBOT']
# get importance
importance = automl.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature:',features[i], 'Score: %.5f' % (v))
# plot feature importance
plt.bar(features,importance)
plt.xticks(rotation=-45)
plt.xlabel('Feature')
plt.ylabel('Feature importance score')

plt.savefig('CatBoost Feature Importance.png', bbox_inches="tight", pad_inches=0.5)

##ONLY NEED TO SAVE THIS ONCE (PER NUMBER OF LAGS), NOT FOR EACH MODEL
#observe_train = df_train.iloc[:,:5]
#print("Saving train to 'train_reg.csv'")
#observe_train.to_csv('train_reg.csv')
