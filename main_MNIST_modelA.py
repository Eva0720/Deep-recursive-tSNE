# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:31:55 2020

@author: Summer
"""


import numpy as np
np.random.seed(71)

import matplotlib
matplotlib.use('Agg')
#from memory_profiler import profile
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input,BatchNormalization,Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.datasets import mnist,fashion_mnist,cifar10
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

#GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#parametric settings
batch_size = 2500
low_dim =2
nb_epoch = 350

shuffle_interval = nb_epoch + 1
#n_jobs = 4

perplexity = 30.0

np.seterr(divide='ignore',invalid='ignore')

def calculate_P(X):
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):
        P_batch = x2p(X[i:i + batch_size],perplexity)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        
                #exaggerate
        P_batch = P_batch*2
                
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

def CEtsne(P, Y):
    alpha = low_dim - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C1 = K.sum(P*K.log((P + eps) / (Q + eps)))
    C2 = K.sum((1-P)*K.log(((1-P) + eps) / ((1-Q) + eps)))
    C=C1+C2
    return C

def CEumap(X, Y):
    a=1.929
    b=0.7915
#    spread=1.0
#    min_dist=0.1
#   a, b = find_ab_params(spread, min_dist)
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + a*D, -(2*b) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)  
    X /= K.sum(X)
    C1 = K.sum(X*K.log((X + eps) / (Q + eps)))
    C2 = K.sum((1-X)*K.log(((1-X) + eps) / ((1-Q) + eps)))
    C=C1+C2
    return C

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
channel=1
n, row, col = X_train.shape
X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train.reshape(-1, channel * row * col)
X_test = X_test.reshape(-1, channel * row * col)
X_train /= 255
X_test /= 255

batch_num = int(n // batch_size)
m = batch_num * batch_size


model = Sequential()

## vector-based model 
model.add(Dense(500, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(2000))
model.add(Activation('relu',name='Dense1'))
model.add(Dense(500))
model.add(Activation('relu',name='Dense2'))
model.add(Dense(100))
model.add(Activation('relu',name='Dense3'))
model.add(Dense(2))

model.compile(loss=KLdivergence, optimizer="adam")


images = []
fig = plt.figure(figsize=(5, 5))
loss_record=[]
start =time.clock()

for epoch in range(nb_epoch):
   ## calculate P in different recursions 
    if epoch % shuffle_interval == 0:
        X = X_train
        low_para=[]
        for i in range(0, n, batch_size):
            low_para1=calculate_P(X[i:i+batch_size])
            low_para.append(low_para1)
    if epoch==100:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense1').output)
        low_para_model_ouput = low_para_model.predict(X_train)
        low_para=[]
        for i in range(0, n, batch_size):
            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1)     
    if epoch==150:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense2').output)
        low_para_model_ouput = low_para_model.predict(X_train)
        low_para=[]
        for i in range(0, n, batch_size):
            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1)  
    if epoch==200:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense3').output)
        low_para_model_ouput = low_para_model.predict(X_train)
        low_para=[]
        for i in range(0, n, batch_size):
            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1) 
    if epoch==250:   
        model.compile(loss=CEumap, optimizer="adam")
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense3').output)
        X1 = low_para_model.predict(X_train)
        low_para=[]
        for i in range(0, n, batch_size):
            test_hv=hd_v(X1[i:i+batch_size])
            low_para1=test_hv.toarray()
            low_para.append(low_para1)            
         
    # train
    loss=0
    temp_lp=0
    for i in range(0, n, batch_size):
        low_para_temp1=low_para[temp_lp]
        loss += model.train_on_batch(X_train[i:i+batch_size], low_para_temp1)
        temp_lp=temp_lp+1
    loss_record.append(loss / batch_num)
    print ("Epoch: {}/{}, loss: {}".format(epoch+1, nb_epoch, loss / batch_num))

## save results generated by DRE with different recursions        
    if epoch==99:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))       
        pred1 = model.predict(X_train)
        colors = ['darkorange', 'deepskyblue', 'gold', 'lime', 'k', 'darkviolet','peru','olive','midnightblue','palevioletred']
        cmap = matplotlib.colors.ListedColormap(colors[::-1])  
        plt.scatter(pred1[:, 0], pred1[:, 1], cmap=cmap,c=np.squeeze(y_train), marker='o', s=1, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_pre_train.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))       
        pred2 = model.predict(X_test)
        plt.scatter(pred2[:, 0], pred2[:, 1], cmap=cmap,c=np.squeeze(y_test), marker='o', s=1.0, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_pre_test.png")
        model.save('/.../mnist_pre.h5')
    if epoch==149:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))
        pred3 = model.predict(X_train)
        plt.scatter(pred3[:, 0], pred3[:, 1], cmap=cmap,c=np.squeeze(y_train), marker='o', s=0.2, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_re1_train.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))       
        pred4 = model.predict(X_test)
        plt.scatter(pred4[:, 0], pred4[:, 1], cmap=cmap,c=np.squeeze(y_test), marker='o', s=1.0, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_re1_test.png")        
        model.save('/.../mnist_re1.h5')
    if epoch==199:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))
        pred5 = model.predict(X_train)
        plt.scatter(pred5[:, 0], pred5[:, 1], cmap=cmap,c=np.squeeze(y_train), marker='o', s=0.2, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_re2_train.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))       
        pred6 = model.predict(X_test)
        plt.scatter(pred6[:, 0], pred6[:, 1],cmap=cmap, c=np.squeeze(y_test), marker='o', s=1.0, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_re2_test.png")        
        model.save('/.../mnist_re2.h5')   
    if epoch==249:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))
        pred7 = model.predict(X_train)
        plt.scatter(pred7[:, 0], pred7[:, 1], cmap=cmap,c=np.squeeze(y_train), marker='o', s=0.2, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_re2_train.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))       
        pred8 = model.predict(X_test)
        plt.scatter(pred8[:, 0], pred8[:, 1],cmap=cmap, c=np.squeeze(y_test), marker='o', s=1.0, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../mnist_re3_test.png")        
        model.save('/.../mnist_re3.h5')          



plt.clf()
fig = plt.figure(figsize=(5, 5))
pred_train = model.predict(X_train)
colors = ['darkorange', 'deepskyblue', 'gold', 'lime', 'k', 'darkviolet','peru','olive','midnightblue','palevioletred']
cmap = matplotlib.colors.ListedColormap(colors[::-1])  
plt.scatter(pred_train[:, 0], pred_train[:, 1],cmap=cmap, c=np.squeeze(y_train),  marker='o', s=0.5, edgecolor='')
fig.tight_layout()
plt.savefig("/.../modela_dre_train.png")
plt.clf()
fig = plt.figure(figsize=(5, 5))       
pred_test = model.predict(X_test)
plt.scatter(pred_test[:, 0], pred_test[:, 1],cmap=cmap, c=np.squeeze(y_test), marker='o', s=1.0, edgecolor='')
fig.tight_layout()
plt.savefig("/.../modela_dre_test.png")        
model.save('/.../mnist_dre.h5')

