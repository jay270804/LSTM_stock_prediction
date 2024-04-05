import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yfin
import pandas_datareader.data as web
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import date



start='2010-01-01'
end=date.today()
yfin.pdr_override()


st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock ticker:','AAPL')
df = web.get_data_yahoo(user_input, start=start, end=end)
# df.head()
# print(df.tail())

#giving description
st.subheader('Data from 2010-2024')
st.write(df.describe())

#Visualisation below
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Days Moving Average')
MA100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,label='Closing Price')
plt.plot(MA100,'r',label='100 Day Moving Avg')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200 Days Moving Average')
MA100=df.Close.rolling(100).mean()
MA200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,label='Closing Price')
plt.plot(MA100,'r',label='100 Day Moving Avg')
plt.plot(MA200,'g',label='200 Day Moving Avg')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# train_data=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# test_data=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
train_data=pd.DataFrame(df['Close'][0:101])
test_data=pd.DataFrame(df['Close'][101:int(len(df))])

#converting this sequential data into 01 values.
scaler=MinMaxScaler(feature_range=(0,1))

    
#loading my already trained model

model=load_model('jay_stocktrend_predictionmodel.h5')


#testing part
past_100_days=train_data.tail(100)
final_df = pd.concat([past_100_days, test_data], ignore_index=True)

#scaling the testing data to 01 format
actual_data=scaler.fit_transform(final_df)


#again making partition of 100 days and value of 101th day but this time for test data
x_test=[]
y_test=[]

for i in range(100,actual_data.shape[0]):
    x_test.append(actual_data[i-100:i])
    y_test.append(actual_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#making predictions now
y_predicted=model.predict(x_test)


#getting the scaling factor
scaler_reverse=scaler.scale_

#converting scaled data 01 back to original values from dataset.
scale_fact=1/scaler_reverse[0]
y_predicted,y_test=(y_predicted*scale_fact),(y_test*scale_fact)

#finally drawing a graph with actual outcome and my bot's predictions.
st.subheader('Predicted Prices vs Original Prices')
fig2=plt.figure(figsize=(12,6))
plt.xticks(np.arange(0, 4000, step=500), labels=['2010','2012','2014','2016','2018','2020','2022','2024'])
plt.plot(y_test,'g',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)