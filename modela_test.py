

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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#from tsne_utils import x2p

#import pandas as pd
#
#from sklearn import metrics
#
#
#
#letters = pd.read_csv('letter-recognition.txt')
#
#training_points = np.array(letters[:15000].drop(['letter'], 1))
#training_labels = np.array(letters[:15000]['letter'])
#test_points = np.array(letters[15000:].drop(['letter'], 1))
#test_labels = np.array(letters[15000:]['letter'])
#
#mapping = {}
#classes = set(training_labels)
#
#for c in classes:
#    if c not in mapping:
#        mapping[c] = len(mapping)
#y_train = np.array([mapping[i] for i in training_labels])
#
#mapping = {}
#classes = set(test_labels)
#
#for c in classes:
#    if c not in mapping:
#        mapping[c] = len(mapping)
#y_test = np.array([mapping[i] for i in test_labels])
#
#X_train=training_points
#X_test=test_points

#from keras.utils import to_categorical
#y = to_categorical(y, len(mapping))
#def noramlization(data):
#    minVals = data.min(0)
#    maxVals = data.max(0)
#    ranges = maxVals - minVals
#    normData = np.zeros(np.shape(data))
#    m = data.shape[0]
#    normData = data - np.tile(minVals, (m, 1))
#    normData = normData/np.tile(ranges, (m, 1))
#    return normData
#
#X_train=noramlization(X_train)
#X_test=noramlization(X_test)

batch_size = 2500
low_dim =2
nb_epoch = 550

shuffle_interval = nb_epoch + 1
n_jobs = 4
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

#        H, thisP = Hbeta(Di, beta)
#        Hdiff = H - logU
#        tries += 1

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


#    expanded = np.expand_dims(X, 1)
#    # "tiled" is now stacked up all the samples along dimension 1
#    tiled = np.tile(expanded, np.stack([1, n, 1]))
#    tiled_trans = np.transpose(tiled, axes=[1,0,2])
#    diffs = tiled - tiled_trans
#    D = np.sum(np.square(diffs), axis=2)


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
X_train1=np.load('/workspace/DRE/MNIST/X.npy')
X_train=X_train1[0:22500]
#color=np.load('E:/RNA_SEQ/color.npy')
color1=np.load('/workspace/DRE/MNIST/color.npy')
color1=color1[0:22500]
color2=np.load('/workspace/DRE/MNIST/color_class.npy')
color2=color2[0:22500]
n=22500
channel=1
batch_num = int(n // batch_size)
m = batch_num * batch_size

#model = Sequential()
#model.add(Embedding(ALPHABET_SIZE, EMB_DIM))
#model.add(Conv1D(64,5, padding='same',activation='relu',strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(32,5, padding='same',activation='relu',strides=1))
#model.add(GlobalAveragePooling1D())
#model.add(Dense(2000))
#model.add(Activation('relu',name='Dense1'))
#model.add(BatchNormalization())
#model.add(Dense(500))
#model.add(Activation('relu',name='Dense2'))
#model.add(BatchNormalization())
#model.add(Dense(100))
#model.add(Activation('relu',name='Dense3'))
#model.add(Dense(2))


model = Sequential()
#model.add(Embedding(ALPHABET_SIZE, EMB_DIM))
## vector-based model
model.add(Dense(500, input_shape=(X_train.shape[1],)))
#model.add(Conv1D(64,5, padding='same',activation='relu',strides=1))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Conv1D(32,5, padding='same',activation='relu',strides=1))
model.add(Dense(500))
#model.add(Conv1D(32, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Dense(500))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Dense(500))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Reshape((2500,48000)))
model.add(Dense(2000))
model.add(Activation('relu',name='Dense1'))
model.add(BatchNormalization())
model.add(Dense(500))
model.add(Activation('relu',name='Dense2'))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('relu',name='Dense3'))
model.add(BatchNormalization())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(2))

## convolution-based model
#model.add(Convolution2D(input_shape=(28,28,1), filters=16, kernel_size=3, strides=1, padding='same',activation = 'relu', name='conv1'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(filters=32, kernel_size=3, strides=1,padding='same', activation = 'relu', name='conv2'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(SelfAttention(ch=32,name='atten'))
#model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu', name='conv3'))
#model.add(Flatten())
##model.add(Dense(500,activation = 'relu'))
##model.add(BatchNormalization())
#model.add(Dense(2000,activation = 'relu',name='Dense1'))
#model.add(Dense(500,activation = 'relu',name='Dense2'))
#model.add(Dense(100,activation = 'relu',name='Dense3'))
### buffer layer that can be considered when using fashion-mnist dataset
##model.add(Dense(100,activation = 'relu',name='Dense4'))
#model.add(Dense(2))
model_name='/storage/TVCG_major2/RNA/3dense/model_vector_re2.h5'
model=load_model(model_name,custom_objects={"KLdivergence":KLdivergence,"SelfAttention":SelfAttention,"CEumap":CEumap})

pred_train = model.predict(X_train)



pred_test=pred_train
#y_test=y_train
X = X_train
#X = X.reshape(-1, 28 * 28 * 1)
#nn_nh = metrics.metric_neighborhood_hit(pred_test, y_test)
nn_tr = metrics.metric_trustworthiness(X, pred_test, k=7)
nn_co = metrics.metric_continuity(X, pred_test, k=7)
nn_sh = metrics.metric_pq_shepard_diagram_correlation(X, pred_test)

from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler
def normalize(D):
    scaler = MinMaxScaler()
    D = scaler.fit_transform(D.reshape((-1, 1)))
    D = D.squeeze()
    return D

def normalized_stress(D_high, D_low):
    D_high=normalize(D_high)
    D_low=normalize(D_low)
    return np.sqrt(np.sum((D_high - D_low)**2) / np.sum(D_high**2))

D_h, D_l = squareform(pdist(X)), squareform(pdist(pred_test))   
D_h=normalize(D_h)
D_l=normalize(D_l) 
nn_s = normalized_stress(D_h, D_l)
#print("nn_nh=%.2f%%" % (nn_nh))
print("nn_tr=%.2f%%" % (nn_tr))
print("nn_co=%.2f%%" % (nn_co))
print("nn_sh=%.2f%%" % (nn_sh))
print("nn_s=%.2f%%" % (nn_s))
## Calculating 1-NN accuracy
#trainLabels = y_train
#testLabels = y_test
#trainData = pred_train
#testData = pred_test
#accuracies = []
#k=1
#model1 = KNeighborsClassifier(n_neighbors=k)
#model1.fit(trainData, trainLabels)
#score = model1.score(testData, testLabels)
#print("k=%d, accuracy=%.2f%%" % (k, score * 100))
## results visualization
#plt.clf()
#fig = plt.figure(figsize=(5, 5))
#pred_test = model.predict(X_test)
#colors = ['darkorange', 'deepskyblue', 'gold', 'lime', 'k', 'darkviolet','peru','olive',
#          'midnightblue','palevioletred','tomato','lawngreen', 'cornflowerblue','slategray',
#          'chocolate','firebrick','rosybrown','olivedrab','darkgray','tan','cyan','indigo',
#          'fuchsia','darkkhaki','teal','darksalmon']
#cmap = matplotlib.colors.ListedColormap(colors[::-1])  
#plt.scatter(pred_test[:, 0], pred_test[:, 1],marker='o',c=np.squeeze(y_test),cmap=cmap,s=0.5, edgecolor='')
#fig.tight_layout()
#
#plt.savefig("test_letter_re3.png")

##save model
#model.save('model_letter_re3.h5') 

## Calculating 1-NN accuracy
#trainLabels = y_train
#testLabels = y_test
#trainData = pred1
#testData = pred2
#accuracies = []
#k=1
#model1 = KNeighborsClassifier(n_neighbors=k)
#model1.fit(trainData, trainLabels)
#score = model1.score(testData, testLabels)
#print("k=%d, accuracy=%.2f%%" % (k, score * 100))

