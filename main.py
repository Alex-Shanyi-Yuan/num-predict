from imp import new_module
from operator import mod
import tensorflow as tf #tensor flow
import matplotlib.pyplot as plt #ploting images
import numpy as np

# #loading set
# mnist = tf.keras.datasets.mnist

# #x contain images, and y contain labels(solution)
# #divide train set and test set
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# #images have black background and letters in white
# #normalize data
# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_test, axis = 1)

# #add more one dimention for kernal operations
# IMG_SIZE = 28
# x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# #create neural network
# model = Sequential()

# #first convolution layer 28+1-3 = 26
# model.add(Conv2D(64,(3,3), input_shape = x_trainr.shape[1:])) #64 entries/filters in layer, of 3by3 images
# model.add(Activation("relu")) #make non-linear (remove 0s)
# model.add(MaxPooling2D(pool_size=(2,2))) #get max 2x2 value goes to next layer

# #second convolution layer 
# model.add(Conv2D(64,(3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2))) 

# #third convolution layer 
# model.add(Conv2D(64,(3,3))) 
# model.add(Activation("relu")) 
# model.add(MaxPooling2D(pool_size=(2,2))) 

# #decision layer 1
# model.add(Flatten()) #2D into 1D
# model.add(Dense(64))
# model.add(Activation("relu"))

# #decision layer 2
# model.add(Dense(32))
# model.add(Activation("relu"))

# #decision layer 3
# model.add(Dense(10)) #must be 10
# model.add(Activation("softmax"))

# #train model
# model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
# model.fit(x_trainr, y_train, epochs=5, validation_split = 0.3)

# #save model
# model.save("neural.h5")

