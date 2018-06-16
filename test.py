
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import math
import matplotlib.pyplot as plt
import numpy
import webbrowser
import os
import time

def funcToLearn(x):
    return x*x-4


def buildModel():
    lr=0.04
    alpha=0.125
    model = Sequential()
    #model.add(Dense(24, activation='tanh',input_shape=[1]))
    #model.add(Dense(40, activation='tanh'))
    #model.add(Dense(24, activation='tanh'))
    model.add(Dense(1, input_shape=[1]))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    model.add(Dense(1))
    model.add(keras.layers.LeakyReLU(alpha=alpha))
    model.add(Dense(1))
    model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.RMSprop(lr=lr,decay=0.0015))
    return model


def getArrs(func,start,end,vals):
    step=end-start
    step/=vals
    inVals=numpy.empty([vals,1],dtype=numpy.float32)
    outVals=numpy.empty([vals,1],dtype=numpy.float32)
    for i in range(vals):
        inVals[i]=start+i*step
        outVals[i]=func(inVals[i])
    return inVals,outVals


def plot(func,start,end,vals):
    inVals, outVals=getArrs(func,start,end,vals)
    plt.figure(0)
    plt.plot(inVals, outVals)
    plt.show()

xxpected,yxpected=getArrs(funcToLearn,0,4,1200)
xtrain,ytrain=getArrs(funcToLearn,0,2.5,1280)

model=buildModel()

for i in range(10):
    print(str(i))
    model.fit(x=xtrain,y=ytrain,epochs=1)
    xtest,ytest=getArrs(model.predict,0,3.2,512)
    plt.figure(i)
    i += 1
    plt.plot(xtest.flatten(), ytest.flatten(), 'b')
    plt.plot(xxpected.flatten(), yxpected.flatten(), 'r')
    plt.plot(xtrain.flatten(), ytrain.flatten(), 'g')
    plt.savefig('plots/plot'+str(i)+'.pdf')
    os.startfile('plots\\plot'+str(i)+'.pdf')#webbrowser.open('plots/plot'+str(i)+'.pdf')

    time.sleep(0.75)



#plot(funcToLearn,-3,3,1000)