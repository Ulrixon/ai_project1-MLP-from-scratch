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


#%% import data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train_matrix = np.zeros((len(y_train), 10))
for i in range(0, len(y_train) - 1):
    y_train_matrix[i, y_train[i]] = 1
y_test_matrix = np.zeros((len(y_test), 10))
for i in range(0, len(y_test) - 1):
    y_test_matrix[i, y_test[i]] = 1

#%%

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation='sigmoid'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(50, activation='sigmoid'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='sigmoid')
])

loss_fn=tf.keras.losses.MeanSquaredError()
sgd = tf.keras.optimizers.SGD(lr=1)
model.compile(optimizer=sgd,
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train,y_train_matrix,validation_data = (x_test, y_test_matrix),epochs=50,verbose=1, batch_size=10,)

#%%
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#%%
pred = model(x_test)
correct=0
for i in range(0, len(x_test)):

    
    
    
    if np.argmax(pred[i]) == y_test[i]:
        correct += 1
# %%
node100_50=np.array([[8175], [8714], [8924] ,[9059] ,[9131]])
node100_50_acc=(node100_50)/10000
node150_100=np.array([[8456] ,[8894] ,[9076] ,[9171] ,[9214]])
node150_100_acc=(node150_100)/10000

x=list(range(1,6))

plt.plot(x,node100_50_acc,label="100-50")
plt.plot(x,node150_100_acc,label="150-100")
plt.legend(loc="upper left")
plt.xticks(range(1,6))
plt.title('Learning Curve',fontsize="18")
plt.xlabel('epochs', fontsize="10") # 設定 x 軸標題內容及大小
plt.ylabel('accuracy', fontsize="10")
plt.show()
# %%
