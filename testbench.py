# -*- coding: utf-8 -*-
from __future__ import print_function
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
#import data
series = pd.read_csv('testing_data.csv',header=None)
print(series[1])

# use AR and fit data, use MSE find the 
#sample delay, which best fit to output

#see ar.fix() for details
#ic: str{'aic','bic','hic','t-stat'},orders, which
#need to be chosen in order to limit the calculation time
# train = [1:10]
# model = AR(train)
# model_fit = model.fit()