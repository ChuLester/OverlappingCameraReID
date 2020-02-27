# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:50:24 2019

@author: user
"""

import cv2
import numpy as np
import pickle
import os
camN = 4
file = open('D:/MTMCT/OpenPose_ResNet/3c_outlet_no_pos.pkl','rb')
Save = pickle.load(file)
file.close()
dict_pos = {}

def predictFoot(box,h):
    if h == 0:
        return np.array([(box[1]+box[3])/2,box[2]])
    else:
        return np.array([box[1]+box[3]/2,box[0]+ 2*(box[4]-box[0])])

for i in range(0,len(Save),1):
    save = Save[i][0]
    Id = save[0]
    bbox = save[2]
    h = save[3]

    Dict = {}
    for c in range(camN):
        person = Id[c]
        boxes = bbox[c]
        half = h[c]
        if person == None:continue
        for p in range(len(person)):
            if person[p] in Dict:
                Dict[person[p]][c] = predictFoot(boxes[p],half[p])
            else:
                null = np.array([-1,-1])
                Dict[person[p]] = [null] * camN                
                Dict[person[p]][c] = predictFoot(boxes[p],half[p])

    for key in Dict.keys():
        if key == None:continue
        if key in dict_pos:
            dict_pos[key].append(Dict[key])
        else:
            dict_pos[key] = [Dict[key]]
# %%
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
n_X = []
n_Y = []     


# for value in dict_pos.values():
value = dict_pos[0]
X = np.array(value)
All = np.array(range(len(X)))
invalid = np.unique(np.where(X==-1)[0])
valid = list(set(All) - set(invalid))
# if valid == []:continue
TrainX = X[valid].reshape(-1,2*camN)

mds = MDS(2)
mds.fit(TrainX)
pos = mds.embedding_
dbscan = DBSCAN(100,1)
dbscan.fit(pos)
label = dbscan.labels_
color = np.random.uniform(size=(1,3))
for i in range(len(pos)):
    if label[i] > -1:
        # plt.scatter(pos[i][0],pos[i][1],c=color)
        n_X.append(TrainX[i])
        n_Y.append(pos[i])




X = np.array(n_X).reshape(-1,2*camN)
mds = MDS(2)
mds.fit(X)
pos = mds.embedding_
color = np.random.uniform(size=(1,3))
for i in range(len(pos)):
    plt.scatter(pos[i][0],pos[i][1],c=color)
Y = pos
# %%
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor

def E(a,b):
    return np.linalg.norm(a-b) 

import pickle

Y = (pos-np.min(pos))/(np.max(pos)-np.min(pos))
X_train, X_test, y_train, y_test = train_test_split(X ,Y, test_size=0.1)
X_train = X
y_train = Y
ErrorX = []
ErrorY = []
ErrorE = []
for i in range(camN):
    train,test = X_train[:,2*i:2*(i+1)],X_test[:,2*i:2*(i+1)]
    mout = MultiOutputRegressor(GradientBoostingRegressor())
    mout.fit(train,y_train)
    print(mout.score(test,y_test))
    Ex = []
    Ey = []
    Ee = []
    for b in range(len(test)):
        box = test[b]
        ppos = y_test[b]
        pred = mout.predict([box])[0]
        distance = E(ppos,pred)
        Ex.append(abs(ppos[0]-pred[0]))
        Ey.append(abs(ppos[1]-pred[1]))
        Ee.append(distance)
        
    ErrorX.append(Ex)
    ErrorY.append(Ey)
    ErrorE.append(Ee)
    file = open(str(i) + '.pickle','wb')
    pickle.dump(mout,file)
    file.close()
    





