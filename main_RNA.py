# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:20:02 2021

@author: Summer
"""

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
import multiprocessing as mp
from keras.models import load_model
from keras.models import Model
import gc
from keras.preprocessing.sequence import pad_sequences
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


batch_size = 2500
low_dim =2
nb_epoch = 550

shuffle_interval = nb_epoch + 1

perplexity = 30.0

np.seterr(divide='ignore',invalid='ignore')

def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p_job(data):
    i, Di, tol, logU = data
    beta = 1.0
    betamin = -np.inf
    betamax = np.inf
    H, thisP = Hbeta(Di, beta)
    Hdiff = 100*tol
#    Hdiff = H - logU
    tries = 0
 #   while np.abs(Hdiff) > tol and tries < 50:
    while tries < 50:
        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries += 1
        if Hdiff > 0:
            betamin = beta
            if betamax == -np.inf:
                beta = beta * 2
            else:
                beta = (betamin + betamax) / 2
        else:
            betamax = beta
            if betamin == -np.inf:
                beta = beta / 2
            else:
                beta = (betamin + betamax) / 2

    return i, thisP


def x2p(X,perplexity):
 #   tol = 1e-5
    tol = 1e-4
    n = X.shape[0]
    logU = np.log(perplexity)
  
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))

    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])

    result=[]
    for i in range(n):
        data_setin=i, D[i], tol, logU
        result1=x2p_job(data_setin)
        result.append(result1)
    P = np.zeros([n, n])
    for i, thisP in result:
        P[i, idx[i]] = thisP
    return P


def calculate_P(X):
#    print ("Computing pairwise distances...")
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


print ("load data")
# # RNA-SEQ
#X_train1=np.load('E:/RNA_SEQ/X.npy')
X_train1=np.load('/.../data/X.npy')
X_train=X_train1[0:22500]
#color=np.load('E:/RNA_SEQ/color.npy')
color1=np.load('/.../data/color.npy')
color1=color1[0:22500]
color2=np.load('/.../data/color_class.npy')
color2=color2[0:22500]
n=22500
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
#model.add(Dense(50))
#model.add(Activation('relu'))
model.add(Dense(2))


model.compile(loss=KLdivergence, optimizer="adam")

   
print ("fit")
images = []
fig = plt.figure(figsize=(5, 5))
loss_record=[]
for epoch in range(nb_epoch):
   ## shuffle X_train and calculate P in different recursions 
    if epoch % shuffle_interval == 0:

        X = X_train
        low_para=[]
        for i in range(0, n, batch_size):
#            low_para1=cal_matrix_P(X[i:i+batch_size],30)
            low_para1=calculate_P(X[i:i+batch_size])
            low_para.append(low_para1)
    if epoch==150:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense1').output)
        low_para_model_ouput = low_para_model.predict(X_train)
        low_para=[]
        for i in range(0, n, batch_size):
#            low_para1=cal_matrix_P(low_para_model_ouput[i:i+batch_size],30)
            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1)
    if epoch==250:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense2').output)
        low_para_model_ouput = low_para_model.predict(X_train)
        low_para=[]
        for i in range(0, n, batch_size):
#            low_para1=cal_matrix_P(low_para_model_ouput[i:i+batch_size],30)
            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1)
    if epoch==350:   
        low_para_model = Model(inputs=model.input,outputs=model.get_layer('Dense3').output)
        low_para_model_ouput = low_para_model.predict(X_train)
        low_para=[]
        for i in range(0, n, batch_size):
#            low_para1=cal_matrix_P(low_para_model_ouput[i:i+batch_size],30)
            low_para1=calculate_P(low_para_model_ouput[i:i+batch_size])
            low_para.append(low_para1)
    if epoch==450:   
  
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
    if epoch==149:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred1 = model.predict(X_train)
        plt.scatter(pred1[:, 0], pred1[:, 1], marker='o', s=0.5, color=color1[0:22500])
        fig.tight_layout()
        plt.savefig("/.../RNA/vector1_train_pre.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred1 = model.predict(X_train)
        plt.scatter(pred1[:, 0], pred1[:, 1], marker='o', s=0.5, color=color2[0:22500])
        fig.tight_layout()
        plt.savefig("/.../RNA/vector2_train_pre.png")
        model.save('/.../RNA/model_vector_pre.h5') 
    if epoch==249:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred3 = model.predict(X_train)
        plt.scatter(pred3[:, 0], pred3[:, 1], marker='o', s=0.5, color=color1[0:22500])
        fig.tight_layout()
        plt.savefig("/.../RNA/vector1_train_re1.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred3 = model.predict(X_train)
        plt.scatter(pred3[:, 0], pred3[:, 1], marker='o', s=0.5, color=color2[0:22500])
        fig.tight_layout()
        plt.savefig("/.../RNA/vector2_train_re1.png")
        model.save('/.../RNA/model_vector_re1.h5')
    if epoch==349:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred5 = model.predict(X_train)
        plt.scatter(pred5[:, 0], pred5[:, 1], marker='o', s=0.5, color=color1[0:22500])
        fig.tight_layout()
        plt.savefig("/storage/DRE_submission/RNA/vector1_train_re2.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred5 = model.predict(X_train)
        plt.scatter(pred5[:, 0], pred5[:, 1], marker='o', s=0.5, color=color2[0:22500])
        fig.tight_layout()
        plt.savefig("/.../RNA/vector2_train_re2.png")
        model.save('/.../RNA/model_vector_re2.h5')
        
    if epoch==449:
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred7 = model.predict(X_train)
        plt.scatter(pred7[:, 0], pred7[:, 1], marker='o', s=0.5, color=color1[0:22500])
        fig.tight_layout()
        plt.savefig("/.../RNA/vector1_train_re3.png")
        plt.clf()
        fig = plt.figure(figsize=(5, 5))      
        pred7 = model.predict(X_train)
        plt.scatter(pred7[:, 0], pred7[:, 1], marker='o', s=0.5, color=color2[0:22500])
        fig.tight_layout()
        plt.savefig("/.../RNA/vector2_train_re3.png")
        model.save('/.../RNA/model_vector_re3.h5')

pred = model.predict(X_train)

plt.clf()
fig = plt.figure(figsize=(5, 5))   
plt.scatter(pred[:, 0], pred[:, 1], marker='o', s=0.5, color=color1[0:22500])
fig.tight_layout()
plt.savefig("/.../RNA/vector1_train_f.png")
plt.clf()
fig = plt.figure(figsize=(5, 5))    
plt.scatter(pred[:, 0], pred[:, 1], marker='o', s=0.5, color=color2[0:22500])
fig.tight_layout()
plt.savefig("/.../RNA/vector2_train_f.png")
##save model
model.save("/.../RNA/model_vector.h5") 

