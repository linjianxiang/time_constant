# -*- coding: utf-8 -*-
from __future__ import print_function
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from pandas import Series
from pandas import read_csv
import pandas as pd
import numpy as np
#import data
# series = pd.read_csv('testing_data.csv',header=None)
# Input = series[0]
# Output = series[1]

# series = pd.read_csv('daily-minimum-temperatures.csv',header=None)
# # split dataset
# X = series.values
# train, test = X[1:len(X)-7], X[len(X)-7:]
# print(train)
# use AR and fit data

# model = AR(train)
# model_fit = model.fit()
# use MSE find the 

#sample delay, which best fit to output
#see ar.fix() for details
#ic: str{'aic','bic','hic','t-stat'},orders, which
#need to be chosen in order to limit the calculation time
# train = [1:10]
# model = AR(train)
# model_fit = model.fit()
# def parser(x):
# 	return pd.datetime.strptime(x, '%Y-%m-%d')
 
series = pd.read_csv('daily-minimum-temperatures.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=lambda x: pd.datetime.strptime(x, '%d/%m/Y'))
print(series.head())
series.plot()
pyplot.show()
model = AR(series)
model_fit = model.fit()