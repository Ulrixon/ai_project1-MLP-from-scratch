#%%
import imp
from random import random
import numpy as np
import matplotlib.pyplot as plt
from pandas import array
import sklearn
import tensorflow as tf
import pandas as oypd

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

#%% example

trainimg = x_train
trainlabel = y_train
nsample = 1


for i in [0, 1, 2]:
    plt.imshow(np.reshape(trainimg[i, :] / 255, (28, 28)), interpolation="nearest")
    print(y_train[i])
    plt.show()


#%% activition funtion
def sig(x):
    return 1 / (1 + np.exp(-x))


#%% layer
import random

# learning rate
lrate = 0.01
epochs = 5
inputlayer = 28 * 28
hiddenlayer_1 = 150
hiddenlayer_2 = 100
outputlayer = 10
datanumber = 60000
testnumber = 10000
# weight_1=np.empty((hiddenlayer_1,784))


def weight_generator(forelayer, nowlayer):
    weight = np.random.uniform(-1, 1, (nowlayer, forelayer))
    return weight


weight_1 = weight_generator(inputlayer, hiddenlayer_1)
weight_2 = weight_generator(hiddenlayer_1, hiddenlayer_2)
weight_output = weight_generator(hiddenlayer_2, outputlayer)
# for i in range(0,hiddenlayer_1-1):
#    for j in range(0,783):
#        weight_1[i,j]= random.random()

#%% modol training
def output_theta(output_1, target):
    theta = (target - output_1) * output_1 * (1 - output_1)
    return theta


def hidden_theta(middleoutput, next_weight, theta_matrix):
    final_theta_matrix = np.empty((len(middleoutput), 1))
    for i in range(0, len(middleoutput)):
        # for j in range(0,len(next_weight)):
        theta = np.dot(theta_matrix.flatten(), next_weight[:, i])
        theta = middleoutput[i] * (1 - middleoutput[i]) * theta.flatten()
        final_theta_matrix[i] = theta
    return final_theta_matrix


def delta_w_generator(forelayer, nowlayer, theta_matrix, nowoutput, learate):
    delta_w_matrix = np.empty((forelayer, nowlayer))
    for j in range(0, forelayer):  # 生出output layer delta w
        for i in range(0, nowlayer):
            delta_w = learate * theta_matrix[j] * nowoutput[i]
            delta_w_matrix[j, i] = delta_w
    return delta_w_matrix


def output_generator(weight1, weight2, weightoutput, data):
    # firstsum1= np.empty((1, hiddenlayer_1))
    firstsum1 = sig(np.dot(weight1, data))
    # print(firstsum1)
    Secondsum1 = sig(np.dot(weight2, firstsum1.flatten()))
    output1 = sig(np.dot(weightoutput, Secondsum1.flatten()))
    # print(output1)# 先算出初始output
    return output1


correct_matrix = np.zeros((epochs, 1))
for z in range(epochs):
    k_loop_para = list(range(0, datanumber))
    random.shuffle(k_loop_para)
    for k in k_loop_para:  # 樣本迴圈
        firstsum = np.empty((1, hiddenlayer_1))
        # for i in range(0,len(x_train)-1):
        firstsum = sig(np.dot(weight_1, x_train[k, :].flatten()))

        Secondsum = sig(np.dot(weight_2, firstsum.flatten()))
        output = sig(np.dot(weight_output, Secondsum.flatten()))  # 先算出初始output
        # print(output)
        theta_output = output_theta(output, y_train_matrix[k])
        theta_hidden2 = hidden_theta(Secondsum, weight_output, theta_output).flatten()
        theta_hidden1 = hidden_theta(firstsum, weight_2, theta_hidden2).flatten()

        delta_w_matrix_2to_output = delta_w_generator(
            outputlayer, hiddenlayer_2, theta_output, Secondsum, lrate
        )
        delta_w_matrix_1to_2 = delta_w_generator(
            hiddenlayer_2, hiddenlayer_1, theta_hidden2, firstsum, lrate
        )
        delta_w_matrix_inputto_1 = delta_w_generator(
            hiddenlayer_1, inputlayer, theta_hidden1, x_train[k, :].flatten(), lrate
        )
        weight_output = weight_output + delta_w_matrix_2to_output
        weight_1 = weight_1 + delta_w_matrix_inputto_1
        weight_2 = weight_2 + delta_w_matrix_1to_2

    correct = 0
    pred = np.zeros((testnumber, 1))
    for i in range(0, testnumber):

        expect_output = output_generator(
            weight_1, weight_2, weight_output, x_test[i, :].flatten()
        )
        # result_matrix[i, ::] = expect_output
        pred[i] = np.argmax(expect_output)
        if pred[i] == y_test[i]:
            correct += 1
    correct_matrix[z] = correct

print("modol training done")
#%% expect

result_matrix = np.empty((testnumber, 10))
pred = np.zeros((testnumber, 1))
MSE = 0


correct = 0
for i in range(0, testnumber):

    expect_output = output_generator(
        weight_1, weight_2, weight_output, x_test[i, :].flatten()
    )
    result_matrix[i, ::] = expect_output
    pred[i] = np.argmax(expect_output)
    if pred[i] == y_test[i]:
        correct += 1
    # print(expect_output)
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test[range(0, testnumber)], pred.transpose().flatten())
print(correct)
print(MSE)


#%% import data search
import os
from os import listdir
from os.path import isfile, join

import cv2

label_folder = []
total_size = 0
data_path = ""

for root, dirts, files in os.walk(data_path):
    for dirt in dirts:
        label_folder.append(dirt)
        total_size += len(files)

print("found", total_size, "files.")
print("folder:", label_folder)
# %% load img train
import numpy as np

base_x_train = []
base_y_train = []

for i in range(len(label_folder)):
    labelPath = data_path + r"\\" + label_folder[i]

    FileName = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]

    for j in range(len(FileName)):
        path = labelPath + r"\\" + FileName[j]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        base_x_train.append(img)
        base_y_train.append(label_folder[i])

x_train = base_x_train
y_train = base_y_train
# when you load already, u should turn back to train model then go down
#%% load img test
total_size = 0
data_path = ""

base_x_test = []

labelPath = data_path + r"\\"

FileName_test = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]

for j in range(len(FileName_test)):
    path = labelPath + r"\\" + FileName_test[j]

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    base_x_test.append(img)

testnumber = len(base_x_test)
pred = np.zeros((testnumber, 1))


# predict
correct = 0
for i in range(0, testnumber):

    expect_output = output_generator(
        weight_1, weight_2, weight_output, x_test[i, :].flatten()
    )

    pred[i] = np.argmax(expect_output)
    if pred[i] == y_test[i]:
        correct += 1

FileName_test.removesuffix(".img")
# export txt file
path_to_file = "/Users/ryan/Downloads/"
with open(path_to_file + "410873001.txt", "w") as g:
    for t in range(testnumber):
        g.write(FileName_test[t] + " " + pred[t] + "\n")
