
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:59:11 2020

@author: Summer
"""

import numpy as np
np.random.seed(71)

import matplotlib
matplotlib.use('Agg')
#from memory_profiler import profile
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization,Embedding, Conv1D,Reshape,GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.datasets import mnist,fashion_mnist,cifar10,imdb
from sklearn.model_selection import train_test_split
import metrics
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import copy
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from SA_layer import *
from umap_utils import *
from tsne_utils import x2p
import multiprocessing as mp
from keras.models import load_model
from keras.models import Model
import gc
import os
from keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
batch_size = 2500
low_dim =2
nb_epoch = 400

shuffle_interval = nb_epoch + 1
n_jobs = 4
perplexity = 30.0

np.seterr(divide='ignore',invalid='ignore')


def calculate_P(X):
    print ("Computing pairwise distances...")
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):
        P_batch = x2p(X[i:i + batch_size],perplexity)
#        print(P_batch)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        
                #exaggerate
        P_batch = P_batch*2
                
        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12) 
        P[i:i + batch_size] = P_batch
    return P

def CEumap(X, Y):
    a=1.929
    b=0.7915
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


def cal_matrix_P(X,neighbors):
    entropy=np.log(neighbors)
    n1,n2=X.shape
    D=np.square(sklearn.metrics.pairwise_distances(X))
    D_sort=np.argsort(D,axis=1)
    P=np.zeros((n1,n1))
    for i in range(n1):
        Di=D[i,D_sort[i,1:]]
        P[i,D_sort[i,1:]]=cal_p(Di,entropy=entropy)
    P=(P+np.transpose(P))/(2*n1)
    P=np.maximum(P,1e-100)
    return P


def cal_p(D,entropy,K=50):
    beta=1.0
    H=cal_entropy(D,beta)
    error=H-entropy
    k=0
    betamin=-np.inf
    betamax=np.inf
    while np.abs(error)>1e-4 and k<=K:
        if error > 0:
            betamin=copy.deepcopy(beta)
            if betamax==np.inf:
                beta=beta*2
            else:
                beta=(beta+betamax)/2
        else:
            betamax=copy.deepcopy(beta)
            if betamin==-np.inf:
                beta=beta/2
            else:
                beta=(beta+betamin)/2
        H=cal_entropy(D,beta)
        error=H-entropy
        k+=1
    P=np.exp(-D*beta)
    P=P/np.sum(P)
    return P


def cal_entropy(D,beta):
    # P=numpy.exp(-(numpy.sqrt(D))*beta)
    P=np.exp(-D*beta)
    sumP=sum(P)
    sumP=np.maximum(sumP,1e-200)
    H=np.log(sumP) + beta * np.sum(D * P) / sumP
    return H

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


X_train=np.load('/.../X_imdb.npy')
y_train=np.load('/.../y_imdb.npy')

n=25000
channel=1
batch_num = int(n // batch_size)
m = batch_num * batch_size


model = Sequential()

## vector-based model
model.add(Dense(500, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(2000))
model.add(Activation('relu',name='Dense1'))
model.add(BatchNormalization())
model.add(Dense(500))
model.add(Activation('relu',name='Dense2'))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('relu',name='Dense3'))
model.add(BatchNormalization())
model.add(Dense(2))


model.compile(loss=KLdivergence, optimizer="adam")

   
loss_record=[]
for epoch in range(nb_epoch):
   ## shuffle X_train and calculate P in different recursions 
    if epoch % shuffle_interval == 0:
        X = X_train[np.random.permutation(n)[:m]]

        low_para=[]
        for i in range(0, n, batch_size):
#            low_para1=calculate_P(X[i:i+batch_size])
            low_para1=cal_matrix_P(X[i:i+batch_size],30)
            low_para.append(low_para1)
    if epoch==151:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense1').output)
        low_para_model_ouput = low_para_model.predict(X)
        low_para=[]
        for i in range(0, n, batch_size):
#            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para1=cal_matrix_P(low_para_model_ouput[i:i+batch_size],30)
            low_para.append(low_para1)
    if epoch==201:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense2').output)
        low_para_model_ouput = low_para_model.predict(X)
        low_para=[]
        for i in range(0, n, batch_size):
            low_para1=cal_matrix_P(low_para_model_ouput[i:i+batch_size],30)
#            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1)
    if epoch==251:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense3').output)
        low_para_model_ouput = low_para_model.predict(X)
        low_para=[]
        for i in range(0, n, batch_size):
            low_para1=cal_matrix_P(low_para_model_ouput[i:i+batch_size],30)
#            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1)
    if epoch==301:   
        model.compile(loss=CEumap, optimizer="adam")
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense3').output)
        X1 = low_para_model.predict(X)
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
        loss += model.train_on_batch(X[i:i+batch_size], low_para_temp1)
        temp_lp=temp_lp+1
    loss_record.append(loss / batch_num)
    print ("Epoch: {}/{}, loss: {}".format(epoch+1, nb_epoch, loss / batch_num))
    if epoch==150:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred1 = model.predict(X_train)
        plt.scatter(pred1[:, 0], pred1[:, 1], c=np.squeeze(y_train), marker='o', s=0.5, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../IMDB/vector_train_pre.png")   
        model.save('/.../IMDB/model_vector_pre.h5') 

    if epoch==200:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred3 = model.predict(X_train)
        plt.scatter(pred3[:, 0], pred3[:, 1], c=np.squeeze(y_train), marker='o', s=0.5, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../IMDB/vector_train_re1.png")  
        model.save('/.../IMDB/model_vector_re1.h5')

    if epoch==250:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred5 = model.predict(X_train)
        plt.scatter(pred5[:, 0], pred5[:, 1], c=np.squeeze(y_train), marker='o', s=0.5, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../IMDB/vector_train_re2.png")
        model.save('/.../IMDB/model_vector_re2.h5')
        
    if epoch==299:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred7 = model.predict(X_train)
        plt.scatter(pred7[:, 0], pred7[:, 1], c=np.squeeze(y_train), marker='o', s=0.5, edgecolor='')
        fig.tight_layout()
        plt.savefig("/.../IMDB/vector_train_re3.png")
        model.save('/.../IMDB/model_vector_re3.h5')
        
#
plt.clf()
fig = plt.figure(figsize=(5, 5)) 
pred9 = model.predict(X_train)
plt.scatter(pred9[:, 0], pred9[:, 1], c=np.squeeze(y_train), marker='o', s=0.5, edgecolor='')
fig.tight_layout()
plt.savefig("/.../IMDB/vector_train_f.png")

##save model
model.save("/.../IMDB/IMDB_model_vector.h5") 
