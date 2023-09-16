# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

##Import libraries
from IPython.display import display ##Need this to use display() on supercomp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle
import random
import time

##Set random seed
random.seed(10)

##Loading in full dataset
##Load in full dataset (31 ensemble members)
df_full = pd.read_parquet('Final full dataset.parquet', engine = 'pyarrow')

print("Full dataset preview:")
display(df_full)
print("Full dataset info:")
display(df_full.info())

##Autoregressive label features
##Create lagged predictor variables
label_col = ['TREFMXAV_U']
label_col_lag3 = [f+"_lag_3" for f in label_col]

def lag_label(df, num_lags):
    for col in label_col:
        for i in range(1, num_lags+1):
            ##Make lagged column variable using shifted values
            df[col+f'_lag_{i}'] = df.groupby(['lat','lon'])[col].shift(i)
            
    df.dropna(inplace=True)

    return df

num_lags = 3 ##Set number of lags
print("Number of lags:", num_lags)

df_full = lag_label(df_full, num_lags)
display(df_full)

##Checking adding 3 lag days has worked correctly - IT HAS!:)
assert np.array_equal(df_full.loc["2006-01-05",:,:][label_col].values,
                      df_full.loc["2006-01-08",:,:][label_col_lag3].values)

##Split into train-test sets
##(Need to be able to grab 'time'. Also will be using 'lat' and 'lon' as predictor variables, so reset whole index)
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

##Create model object
reg = LinearRegression()
##Fit to X_train, y_train
start = time.time()
reg.fit(X_train, y_train)
stop = time.time()

##Linear Regression does not have any hyperparameters to optimise

##Perform 10-fold cross-validation
##FUNCTION Prints the cross-val score to the screen
def print_cv_scores(cvs):
    # :0.2f prints with 2 decimal places
    print(f"{len(cvs)}-fold CV score: {cvs.mean():0.5f} (+/- {cvs.std()*2:0.5f})")

##Saving our trained model
print("Saving best Autoregression-3 Linear Regression model as '3-Autoreg_linear.pickle' file")
pickle.dump(reg, open("3-Autoreg_linear.pickle", "wb"))
##Saving model coefficients
print("Linear Regression model coefficients:")
display(reg.coef_)
print("Linear Regression intercept:")
display(reg.intercept_)

##Cross-validation score
cvs = cross_val_score(reg, X_train, y_train, cv=10)
print_cv_scores(cvs)

##Print training time
print(f"Training time: {stop - start}s")

##Make predictions
y_pred = reg.predict(X_test)
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
print("Saving test & y_pred column to 'y_pred_3-autoreg_linear.csv'")
observe_test.to_csv('y_pred_3-autoreg_linear.csv')

##ONLY NEED TO SAVE THIS ONCE (PER NUMBER OF LAGS), NOT FOR EACH MODEL
#observe_train = df_train.iloc[:,:5]
#print("Saving train to 'train_label_autoreg-3.csv'")
#observe_train.to_csv('train_label_autoreg-3.csv')