#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:27:30 2019

@author: zzhou2
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:50:17 2019

@author: zzhou2
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:30:15 2019

@author: zzhou2
"""

import numpy as np

np.random.seed(71)

import matplotlib

matplotlib.use('Agg')
# from memory_profiler import profile
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, BatchNormalization, Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.datasets import mnist, fashion_mnist, cifar10
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from keras import applications
from sklearn.neighbors import KNeighborsClassifier
import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from SA_layer import *
import multiprocessing as mp
from keras.models import load_model
from keras.models import Model
import gc
from tsne_utils import x2p
from umap_utils import *
from scipy.sparse import coo_matrix
import time
import os

# GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parametric settings
batch_size = 2500
low_dim = 2
nb_epoch = 500
shuffle_interval = nb_epoch + 1
perplexity = 30.0


def calculate_P(X):
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):
        P_batch = x2p(X[i:i + batch_size], perplexity)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T

        # exaggerate
        P_batch = P_batch * 2

        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12)
        P[i:i + batch_size] = P_batch
    return P


def KLdivergence(P, Y):
    alpha = low_dim - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


def CEumap(X, Y):
    a = 1.929
    b = 0.7915
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + a * D, -(2 * b) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    X /= K.sum(X)
    C1 = K.sum(X * K.log((X + eps) / (Q + eps)))
    C2 = K.sum((1 - X) * K.log(((1 - X) + eps) / ((1 - Q) + eps)))
    C = C1 + C2
    return C


print("load data")

##load data
## mnist
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
n, row, col = X_train.shape
channel = 1
X_train = X_train.reshape(60000, 32, 32, 3)
X_test = X_test.reshape(10000, 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

batch_num = int(n // batch_size)
m = batch_num * batch_size

print("build model")
model = Sequential()
model.add(
    Convolution2D(input_shape=(32, 32, 3), filters=16, kernel_size=3, strides=1, padding='same', activation='relu',
                  name='conv1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(SelfAttention(ch=32,name='atten'))
model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='conv33'))
model.add(Flatten())
model.add(Dense(2048, activation='relu', name='Dense1'))
model.add(Dense(512, activation='relu', name='Dense2'))
model.add(Dense(256, activation='relu', name='Dense3'))
model.add(Dense(128))

model.compile(loss=KLdivergence, optimizer="adam")

## calculate probability distribution in high-dim space
images = []
fig = plt.figure(figsize=(5, 5))
loss_record = []
for epoch in range(nb_epoch):
    ##calculate original P with mini-batch technique
    if epoch % shuffle_interval == 0:
        X = X_train[np.random.permutation(n)[:m]]
        X1 = X.reshape(-1, channel * row * col)
        #        X = X_train
        mini_P = []
        for i in range(0, n, batch_size):
            mini_P1 = calculate_P(X1[i:i + batch_size])
            mini_P.append(mini_P1)

    ##calculate new P in different recursions
    # if epoch == 151:
    #     low_para_model = Model(inputs=model.input, outputs=model.get_layer('Dense1').output)
    #     low_para_model_ouput = low_para_model.predict(X)
    #     mini_P = []
    #     for i in range(0, n, batch_size):
    #         mini_P1 = calculate_P(low_para_model_ouput[i:i + batch_size])
    #         mini_P.append(mini_P1)
    # if epoch == 201:
    #     low_para_model = Model(inputs=model.input, outputs=model.get_layer('Dense2').output)
    #     low_para_model_ouput = low_para_model.predict(X)
    #     mini_P = []
    #     for i in range(0, n, batch_size):
    #         mini_P1 = calculate_P(low_para_model_ouput[i:i + batch_size])
    #         mini_P.append(mini_P1)
    # if epoch == 251:
    #     low_para_model = Model(inputs=model.input, outputs=model.get_layer('Dense3').output)
    #     low_para_model_ouput = low_para_model.predict(X)
    #     mini_P = []
    #     for i in range(0, n, batch_size):
    #         mini_P1 = calculate_P(low_para_model_ouput[i:i + batch_size])
    #         mini_P.append(mini_P1)
    # if epoch == 301:
    #     model.compile(loss=CEumap, optimizer="adam")
    #     low_para_model = Model(inputs=model.input, outputs=model.get_layer('Dense3').output)
    #     low_para_model_ouput = low_para_model.predict(X)
    #     low_para = []
    #     for i in range(0, n, batch_size):
    #         test_hv = hd_v(low_para_model_ouput[i:i + batch_size])
    #         mini_P1 = test_hv.toarray()
    #         mini_P.append(mini_P1)
    ## train DNN
    loss = 0
    temp_lp = 0
    for i in range(0, n, batch_size):
        low_para_temp1 = mini_P[temp_lp]
        loss += model.train_on_batch(X[i:i + batch_size], low_para_temp1)
        temp_lp = temp_lp + 1
    loss_record.append(loss / batch_num)
    print("Epoch: {}/{}, loss: {}".format(epoch + 1, nb_epoch, loss / batch_num))


pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
np.save('/home/zixia/flattened_data/dre_train_128.npy',pred_train)
np.save('/home/zixia/flattened_data/dre_test_128.npy',pred_test)
model.save('/home/zixia/flattened_data/model_dre.h5')

