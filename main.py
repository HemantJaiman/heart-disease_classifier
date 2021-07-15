

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score


data = pd.read_csv('heart.csv')

train = data[:250]
test = data[250:]

train_x  = train.iloc[:,:-1]
train_y  = train.iloc[:,-1]

test_x  = test.iloc[:,:-1]
test_y  = test.iloc[:,-1]





#unsing ANN
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation

#adding input layer and first hidden layer
classifier = Sequential()
classifier.add(Dense(units = 16,activation='relu',kernel_initializer ='uniform',input_dim = 13 ))
#adding second hidden layer
classifier.add(Dense(units = 16,activation='relu',kernel_initializer='uniform'))
#adding output layer
classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))

#compiling layers
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

#fitting ANN to training set
classifier.fit(train_x,train_y,batch_size=10,epochs=250)
#prediction with ANN
pred = classifier.predict(test_x)

pred = (pred>0.5)  #in true or false

#checking accuracy
cm = confusion_matrix(pred,test_y)
print(cm)
acc=accuracy_score(pred,test_y)
print(acc*100)