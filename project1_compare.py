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


# %%
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
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32,(5,5),input_shape=(28,28,1), activation='relu', padding="same"),
  tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  ),
  tf.keras.layers.Conv2D(64,(5,5), activation='relu', padding="same"),
  tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  ),
  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu',kernel_initializer='random_normal', bias_initializer=initializer1),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(50, activation='relu',kernel_initializer='random_normal', bias_initializer=initializer2),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='random_normal', bias_initializer=initializer3)
])

loss_fn=tf.keras.losses.MeanSquaredError()

model.compile(optimizer="adam",
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train,y_train_matrix,validation_data = (x_test, y_test_matrix),epochs=500,verbose=1, batch_size=10000)

#%%
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
from tensorflow.keras.models import Model
model1 = Model(inputs=model.inputs , outputs=model.layers[1].output)
image=x_train[1,:,:,:].reshape(-1,28,28,1)
features = model1.predict(image)
fig = plt.figure(figsize=(28,28))
# %%
for i in range(1,features.shape[3]+1):

    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
    
plt.show()
# %%
