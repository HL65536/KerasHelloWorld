
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import math
import matplotlib.pyplot as plt
import numpy
import webbrowser
import os

def funcToLearn(x):
    return math.sin(x*3)*(x*x)/(x*x+1) #(x*x*x+5*x*x*x*x-x*20)/math.log(x*x+2,2)




def buildModel():
    lr=0.04
    model = Sequential()
    model.add(Dense(24, activation='tanh',input_shape=[1]))
    model.add(Dense(40, activation='tanh'))
    model.add(Dense(24, activation='tanh'))
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

xxpected,yxpected=getArrs(funcToLearn,-4,4,1200)
xtrain,ytrain=getArrs(funcToLearn,-2.5,2.5,65536)

model=buildModel()

for i in range(128):
    print(str(i))
    model.fit(x=xtrain,y=ytrain,epochs=1)
    xtest,ytest=getArrs(model.predict,-3.2,3.2,1024)
    plt.figure(i)
    i += 1
    plt.plot(xtest.flatten(), ytest.flatten(), 'b')
    plt.plot(xxpected.flatten(), yxpected.flatten(), 'r')
    plt.plot(xtrain.flatten(), ytrain.flatten(), 'g')
    plt.savefig('plots/plot'+str(i)+'.pdf')

    os.startfile('plots\\plot'+str(i)+'.pdf')#webbrowser.open('plots/plot'+str(i)+'.pdf')



#plot(funcToLearn,-3,3,1000)