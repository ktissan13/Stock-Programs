# Stock Prediction
# Tissan Kugathas
# May 7 2020

# Python Libraries
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pandas.tseries import converter
import pandas as pd
import numpy as np
import pandas_datareader as web
import math

converter.register()

# static variables
stock_symbol = 'JWCA-H.V'
start_date = '2020-04-03'

# Import Libraries
plt.style.use('fivethirtyeight')

# Get the stock quotes
df = web.DataReader(stock_symbol, data_source='yahoo', start=start_date,
                    end=datetime.today().strftime('%Y-%m-%d'))

# Print the stock data
# print(df)

# Get the numbers of rows and columns in the data set
# print(df.shape)

# Visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price CAD ($)', fontsize=18)

# Show graph
# print(plt.show())

# Create a new dataframe with only the Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model
training_data_len = math.ceil(len(dataset) * .8)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i: 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()
