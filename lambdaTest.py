
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import math
import matplotlib.pyplot as plt
import numpy






def test(x):
    x*=0.00390625
    return x

def buildModel():
    model = Sequential()
    model.add(keras.layers.MaxPooling1D(pool_size=1,strides=1,input_shape=[32,32]))
    #model.add(keras.layers.Lambda(lambda x: test(x),input_shape=[1]))
    model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.SGD(lr=1))
    return model

megs=512
print('preAlloc numpy')
testArr=numpy.zeros(shape=[megs*1024,32,32])#numpy.array([0,20,200,255],dtype=numpy.uint8)
print('postAlloc numpy')
#testArr.shape=[testArr.shape[0],1]
model=buildModel()
result=model.predict(testArr,batch_size=1024,verbose=True)
print(result)
