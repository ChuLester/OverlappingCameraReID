# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:29:54 2020

@author: user
"""

from os import listdir
import numpy as np
import random
rootpath = '../DataSet/mask_train_v2/'
Train = listdir(rootpath)
SampleNum = 100000
Set = []
for i in range(1,len(Train),1):
    Set.append(listdir(rootpath+Train[i]))
    
train = []

for i in range(SampleNum):
    a = random.randint(1,len(Train)-2)
    b = random.randint(1,len(Train)-2)
    while a == b:
        a = random.randint(1,len(Train)-2)
        b = random.randint(1,len(Train)-2)
    
    archor = random.randint(1,len(Set[a])-1)
    positive = random.randint(1,len(Set[a])-1)
   
    while archor == positive:
        archor = random.randint(1,len(Set[a])-1)
        positive = random.randint(1,len(Set[a])-1)        
    
    negetive = random.randint(1,len(Set[b])-1)    
    A = '%s%s/%s'%(rootpath,Train[a+1],Set[a][archor])
    P = '%s%s/%s'%(rootpath,Train[a+1],Set[a][positive])
    N = '%s%s/%s'%(rootpath,Train[b+1],Set[b][negetive])  
    tmp = [A,P,N]
    train.append(tmp)
    
O = []
import cv2
for E in train:
    o = []
    for e in E:
        image = cv2.imread(e)
        image = cv2.resize(image,(128,256))
        o.append(image)
    O.append(o)
    
O = np.array(O)
Anchor = O[:,0]
Positive = O[:,1]
Negative = O[:,2]

np.save('Anchor.npy',Anchor)
np.save('Positive.npy',Positive)
np.save('Negative.npy',Negative)