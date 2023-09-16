# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##Import relevant libraries
import numpy as np
import pandas as pd
from flaml import AutoML
from IPython.display import display ##Need this to use display() on supercomp
from sklearn.metrics import mean_squared_error
import pickle
import random

##Set random seed
random.seed(10)

##Load in full dataset (31 ensemble members)
df_full = pd.read_parquet('Final full dataset.parquet', engine = 'pyarrow')

print("Full dataset preview:")
display(df_full)
print("Full dataset info:")
display(df_full.info())

##Create lagged predictor variables
##MUST define feature_columns outside of function
#feature_cols = ["TREFHT"]        ##if using 1800 time budget
feature_cols = ["FSNS", "TREFHT"] ##if using 3600 time budget
feature_cols_lag = [f+"_lag_1" for f in feature_cols]

##FUNCTION creates X number of lag columns for predictor variables
def lag_pred(df, num_lags):
    for col in feature_cols:
        for i in range(1, num_lags+1):
            ##Make lagged column variable using shifted values
            df[col+f'_lag_{i}'] = df.groupby(['lat','lon'])[col].shift(i)
            
    df.dropna(inplace=True)
    
    return df

##Set number of lags
num_lags = 1 #int
print("Number of lags:", num_lags)
df_full = lag_pred(df_full, num_lags)

##Checking adding 1 lag-day has worked correctly
assert np.array_equal(df_full.loc["2006-01-03",:,:][feature_cols].values,
                      df_full.loc["2006-01-04",:,:][feature_cols_lag].values)

##Split into train-test sets
##(Need to be able to grab 'time'. Also will be using 'lat' and 'lon' as predictor variables, so reset whole index)
df_full.reset_index(inplace=True)
print("Full dataset with lags preview:")
display(df_full)

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
    "estimator_list": ['rf'],
    "early_stop": True,
    "eval_method": "cv",
    "n_splits": 10,
    "seed": 10  #42, 10, 25
}

##Train model
automl.fit(X_train=X_train, y_train=y_train, verbose=0, **automl_settings)

##Saving our trained model
print("Saving best Autoregression-1 Random Forest model as 'Autoreg-1_rf.pickle' file")
pickle.dump(automl, open("Autoreg-1_rf.pickle", "wb"))

##Best estimator
print('Best Random Forest learner:', automl.best_estimator)
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
##Saving just the important parts of df_test: time, location, ebsemble number
observe_test = df_test.iloc[:,:5]
##Adding y_pred as new column
observe_test['y_pred'] = y_pred

display(observe_test)
print("Saving test & y_pred column to 'y_pred_autoreg-1_rf.csv'")
observe_test.to_csv('y_pred_autoreg-1_rf.csv')

##ONLY NEED TO SAVE THIS ONCE (PER NUMBER OF LAGS), NOT FOR EACH MODEL
#observe_train = df_train.iloc[:,:5]
#print("Saving train to 'train_reg.csv'")
#observe_train.to_csv('train_autoreg-1.csv')
