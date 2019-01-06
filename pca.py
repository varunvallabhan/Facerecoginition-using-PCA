#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:04:27 2018

@author: byakuya
"""
import cv2
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

name=sorted(glob.glob('train database source path /*.png'))
train = []
for filename in name: 
    im=cv2.imread(filename,0)
    train.append(im)
nametest=sorted(glob.glob('test database source path /*.png'))

test = []
for filename1 in nametest:
    im=cv2.imread(filename1,0)
    test.append(im)

X=[]
for x in train:
    x=x/np.linalg.norm(x)
    
    X.append(x.flatten())

col=np.asmatrix(X).T
col-= np.mean(col,axis=0)

cov_mat=np.dot(col.T,col)
eig_values, eig_vec=np.linalg.eig(cov_mat)
X=[]
for x in test:
    x=x/np.linalg.norm(x)
    
    X.append(x.flatten())
coltest=np.asmatrix(X).T
trainY=[]
for x in range(30):
    for y in range(21):
        trainY.append(x+1)

trainY=np.asmatrix(trainY)


idx = eig_values.argsort()[::-1]
eigen_values = eig_values[idx]
eigen_vectors = eig_vec[:, idx]
acc=[]
for r in range(1,15):
    
    proj=np.asmatrix(eigen_vectors[:, :r])
    
    
    
    WE=np.dot(col,proj)
    norm_proj=WE/np.linalg.norm(WE,axis=0)
    
    Y_test=np.dot(norm_proj.T,coltest)
    Y_train=np.dot(norm_proj.T,col)
    
    clas=KNeighborsClassifier(n_neighbors=1)
    clas.fit(Y_train.T,trainY.T)
    predicty=clas.predict(Y_test.T)
    acc.append(accuracy_score(trainY.T, predicty)*100)
    print accuracy_score(trainY.T, predicty)*100
