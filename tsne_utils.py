# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:10:39 2020

@author: Summer
"""
import numpy as np
import sklearn

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

    Hdiff = H - logU
    tries = 0
 #   while np.abs(Hdiff) > tol and tries < 50:
    while tries < 50:
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

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries += 1

    return i, thisP


def x2p(X,perplexity):
    tol = 1e-5
    n = X.shape[0]
    logU = np.log(perplexity)
#    sum_X = np.sum(np.square(X), axis=1)
#    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))
    D=np.square(sklearn.metrics.pairwise_distances(X,metric='euclidean'))
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