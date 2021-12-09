import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st


st.title ("PETROL PRICE TREND PREDICTION")

df = pd.read_csv(r"train_data.csv")

#DESCRIBING DATA
st.subheader("Data Decription")
df =df.dropna()
st.write(df.describe())
df1=df.reset_index()['Petrol (USD)']
st.header("Petrol Price vs Time Series ")
a = plt.figure(figsize=(12,6))
plt.plot(df1,color = 'red',label='price')
plt.legend()
st.pyplot(a)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

time_step = 200
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = load_model("keras_model.h1")

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


### Calculate RMSE performance metrics
st.title('Root Mean Squared Error :')
st.header('RMSE for train data')
import math
from sklearn.metrics import mean_squared_error
b=math.sqrt(mean_squared_error(y_train,train_predict))
st.write(b)
### Test Data RMSE
st.header('RMSE for test data')
c = math.sqrt(mean_squared_error(ytest,test_predict))
st.write(c)



##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting 
# shift train predictions for plotting
look_back=200
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
st.header("Petrol Price Baseline and Predictions")
fig = plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1),color='red',label='orginal data')
plt.plot(trainPredictPlot,color='black',label ='predicted for train data')
plt.plot(testPredictPlot,label='predicted for test data')
plt.show()
plt.legend()
st.pyplot(fig)

x_input=test_data[84:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 weeks


lst_output=[]
n_steps=200
i=0
while(i<30):
    
    if(len(temp_input)>200):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} week input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} week output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
day_new=np.arange(1,201)
day_pred=np.arange(201,231)
st.header("Petrol Price Trend Prediction For Next 10 Weeks")
fig2= plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[611:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output),label='predection for next 10 weeks')
plt.legend()
st.pyplot(fig2)
