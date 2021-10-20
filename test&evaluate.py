

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
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 2500
low_dim =2
perplexity = 30.0

np.seterr(divide='ignore',invalid='ignore')

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

def normalize(D):
    scaler = MinMaxScaler()
    D = scaler.fit_transform(D.reshape((-1, 1)))
    D = D.squeeze()
    return D

def normalized_stress(D_high, D_low):
    D_high=normalize(D_high)
    D_low=normalize(D_low)
    return np.sqrt(np.sum((D_high - D_low)**2) / np.sum(D_high**2))

## load trained model 
model_name='/.../modelB_dre.h5'
model=load_model(model_name,custom_objects={"KLdivergence":KLdivergence,"CEumap":CEumap})


##load data
## mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
n, row, col = X_train.shape
channel = 1
X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

## directly generate embedding results with the trained model
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

X = X_test.reshape(-1, 28 * 28 * 1)
nn_nh = metrics.metric_neighborhood_hit(pred_test, y_test)
nn_tr = metrics.metric_trustworthiness(X, pred_test, k=7)
nn_co = metrics.metric_continuity(X, pred_test, k=7)
nn_sh = metrics.metric_pq_shepard_diagram_correlation(X, pred_test)


D_h, D_l = squareform(pdist(X)), squareform(pdist(pred_test))   
D_h=normalize(D_h)
D_l=normalize(D_l) 
nn_s = normalized_stress(D_h, D_l)
print("nn_nh=%.2f%%" % (nn_nh))
print("nn_tr=%.2f%%" % (nn_tr))
print("nn_co=%.2f%%" % (nn_co))
print("nn_sh=%.2f%%" % (nn_sh))
print("nn_s=%.2f%%" % (nn_s))

## Calculating 1-NN accuracy
trainLabels = y_train
testLabels = y_test
trainData = pred_train
testData = pred_test
accuracies = []
k=1
model1 = KNeighborsClassifier(n_neighbors=k)
model1.fit(trainData, trainLabels)
score = model1.score(testData, testLabels)
print("k=%d, accuracy=%.2f%%" % (k, score * 100))

## results visualization
plt.clf()
fig = plt.figure(figsize=(5, 5))
plt.scatter(pred_train[:, 0], pred_train[:, 1], c=np.squeeze(y_train), marker='o', s=0.2, edgecolor='')
fig.tight_layout()
plt.savefig("/.../train_modelB_dre.png")
plt.clf()
fig = plt.figure(figsize=(5, 5))       
plt.scatter(pred_test[:, 0], pred_test[:, 1], c=np.squeeze(y_test), marker='o', s=1.0, edgecolor='')
fig.tight_layout()
plt.savefig("/.../test_modelB_dre.png")   
