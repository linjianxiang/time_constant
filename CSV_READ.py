import pandas as pd
import numpy as np
from statsmodels.tsa.arx_model import ARX

# df = pd.read_csv('testing_data.csv',index_col=None)
# print(df.values)
# df.index = pd.to_datetime(df.index)
# print(df.index)
# a = df.iloc[1:len(df)]['output']
# b = df.iloc[1:len(df)]['input']
# print(a.index)
# print(b.values)

# series = pd.read_csv('daily-minimum-temperatures.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=lambda x: pd.datetime.strptime(x, '%m/%d/%Y'))

def get_input_output(name):
    df = pd.read_csv(name,index_col=None)
    print(df.values)
    df.index = pd.to_datetime(df.index)
    print(df.index)
    output_data = df.iloc[1:len(df)]['output']
    input_data = df.iloc[1:len(df)]['input']
    return [output_data,input_data]

filename = 'testing_data.csv'
value = get_input_output(filename)
input = value[1]
output = value[0]
model = ARX(output,input)
model_fit = model.fit()
delay = model.select_order(maxlag = 20, ic = 'aic', trend = 'c',method = 'cmle')
print(delay)