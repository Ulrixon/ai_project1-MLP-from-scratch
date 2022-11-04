#%%
import imp
from random import random
from re import S
import numpy as np
import matplotlib.pyplot as plt
from pandas import array
import sklearn
import tensorflow as tf
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils.vis_utils import plot_model


#%% import data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test / 255
x_test = x_test.reshape(-1, 28, 28, 1)
y_train_matrix = np.zeros((len(y_train), 10))
for i in range(0, len(y_train) - 1):
    y_train_matrix[i, y_train[i]] = 1
y_test_matrix = np.zeros((len(y_test), 10))
for i in range(0, len(y_test) - 1):
    y_test_matrix[i, y_test[i]] = 1

#%%
initializer1 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
initializer2 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
initializer3 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model_input = Input(shape=(28,28,1))
conv1=tf.keras.layers.Conv2D(32,(5,5),input_shape=(28,28,1), activation='relu', padding="same")(model_input)
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv1)
conv2=tf.keras.layers.Conv2D(64,(5,5),input_shape=(28,28,1), activation='relu', padding="same")(pool1)
pool2=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv2)
flatten1= tf.keras.layers.Flatten()(pool2)

conv3=tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1), activation='relu', padding="same")(model_input)
pool3=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv3)
flatten2= tf.keras.layers.Flatten()(pool3)

concate=tf.keras.layers.Concatenate(axis=1)([flatten1,flatten2])

dense1=tf.keras.layers.Dense(100, activation='relu',kernel_initializer='random_normal',
            bias_initializer=initializer1)(concate)
dropout1=tf.keras.layers.Dropout(0.2)(dense1)
dense2=tf.keras.layers.Dense(50, activation='relu',kernel_initializer='random_normal', 
                bias_initializer=initializer2)(dropout1)
dropout2=tf.keras.layers.Dropout(0.2)(dense2)
dense3=tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='random_normal',
 bias_initializer=initializer3)(dropout2)
model = Model(inputs=model_input, outputs=dense3)

loss_fn=tf.keras.losses.MeanSquaredError()



plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer="adam",
              loss=loss_fn,
              metrics=['accuracy'])
# %%
model.fit(x_train,y_train_matrix,validation_data = (x_test, y_test_matrix),epochs=500,verbose=1, batch_size=1000)

#%%
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()