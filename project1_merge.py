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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from PIL import Image
from tensorflow.python.keras.layers import Dense, BatchNormalization
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



datagen = ImageDataGenerator(
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  shear_range = 0.1,
  zoom_range=0.1,
  rotation_range = 30
)

xnew=   datagen.flow(x_train, y_train_matrix,batch_size=1000)


#%%






initializer1 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
initializer2 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
initializer3 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model_input = Input(shape=(28,28,1))
#shift=tf.keras.layers.RandomTranslation(0.1,0.1,fill_mode="constant")(model_input)
#rotate=tf.keras.layers.RandomRotation(factor=(0.1),fill_mode="constant")(shift)
#zoom=tf.keras.layers.RandomZoom(0.1,fill_mode="constant")(rotate)
#shear=tf.keras.preprocessing.image.random_shear(0.1,fill_mode='nearest')(zoom)
#a = tf.keras.layers.BatchNormalization()(model_input, training=True)
conv1=tf.keras.layers.Conv2D(32,(5,5),input_shape=(28,28,1), activation='relu', padding="same")(model_input)
pool1=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv1)
#b = tf.keras.layers.BatchNormalization()(pool1, training=True)
conv2=tf.keras.layers.Conv2D(64,(5,5), activation='relu', padding="same")(model_input)
pool2=tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=None, padding="valid", data_format=None,
  )(conv2)
flatten1= tf.keras.layers.Flatten()(pool2)

conv3=tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1), activation='relu', padding="same")(model_input)
pool3=tf.keras.layers.MaxPooling2D(
    pool_size=(4, 4), strides=None, padding="valid", data_format=None,
  )(conv3)

flatten2= tf.keras.layers.Flatten()(pool3)



#conv4=tf.keras.layers.Conv2D(32,(7,7), activation='relu', padding="same")(model_input)

#pool4=tf.keras.layers.MaxPooling2D(
#    pool_size=(4, 4), strides=None, padding="valid", data_format=None,
#  )(conv4)
#flatten3=tf.keras.layers.Flatten()(pool4)

concate=tf.keras.layers.Concatenate(axis=1)([flatten1,flatten2])

dense1=tf.keras.layers.Dense(400, activation='relu',kernel_initializer='random_normal',
            bias_initializer=initializer1)(concate)
dropout1=tf.keras.layers.Dropout(0.2)(dense1)
dense2=tf.keras.layers.Dense(200, activation='relu',kernel_initializer='random_normal', 
                bias_initializer=initializer2)(dropout1)
dropout2=tf.keras.layers.Dropout(0.2)(dense2)
dense3=tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='random_normal',
 bias_initializer=initializer3)(dropout2)
model = Model(inputs=model_input, outputs=dense3)

loss_fn=tf.keras.losses.MeanSquaredError()

adam =tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    
)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer=adam,
              loss=loss_fn,
              metrics=['accuracy'])



# %%
model.fit(x_train,y_train_matrix,
  validation_data = (x_test, y_test_matrix),epochs=100,verbose=1,batch_size=1000,shuffle=True, use_multiprocessing=True,workers=8)

#%%

model.fit(x=xnew,
  validation_data = (x_test, y_test_matrix),epochs=100,verbose=1,batch_size=1000,shuffle=True, use_multiprocessing=True,workers=8)

#%%
model.save('partly_trained.h5')


#%%
from keras.models import load_model
model = load_model('partly_trained.h5')
model.fit(x = datagen.flow(x_train.prefetch(), y_train_matrix.prefetch(),batch_size=1000),
  validation_data = (x_test.prefetch(), y_test_matrix.prefetch()),epochs=500,verbose=1)

#%%
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
plt.imshow(np.reshape(x_train[0, :] / 255, (28, 28)), interpolation="nearest")
print(y_train[0])
plt.show()

plt.imshow(np.reshape(datagen.flow(x_train[0, :].reshape(-1, 28, 28, 1)), (28, 28)), interpolation="nearest")
print(y_train[0])
plt.show()
# %%
datagen = ImageDataGenerator(
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  shear_range = 0.1,
  zoom_range=0.1,
  rotation_range = 30
)
#准备
for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
# grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.show()
    break



# %%
