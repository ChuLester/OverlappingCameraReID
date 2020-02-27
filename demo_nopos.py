# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:35:45 2019

@author: user
"""


import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from GenerateModel import load_model
from Network import network_nopos

w,h = 360,288 #video Resoluation
estimator = TfPoseEstimator(get_graph_path('mobilenet_thin'),target_size=(w,h))
reid = load_model('./model/AMSoftmax_0.05.h5')
Net = network_nopos(1,40,100)
Need_Point = [16,17,2,1,5,8,11,10,13]

cap = []
cap.append(cv2.VideoCapture('D:/mutiple camera/EPFL/Laboratory - 4/4p-c0.avi'))
cap.append(cv2.VideoCapture('D:/mutiple camera/EPFL/Laboratory - 4/4p-c1.avi'))
cap.append(cv2.VideoCapture('D:/mutiple camera/EPFL/Laboratory - 4/4p-c2.avi'))
cap.append(cv2.VideoCapture('D:/mutiple camera/EPFL/Laboratory - 4/4p-c3.avi'))

# %%
n = 1
while True:
    X = []
    abox = []
    ax = []
    T = []
    frames = []
    aah = []
    for cam in range(len(cap)):
        x = []
        ah = []
        pos = []
        box = []
        ret,Ori = cap[cam].read()
        if ret==False:break

        frames.append(Ori)
        result = Ori.copy()

        image = Ori.copy()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        humans = estimator.inference(image,resize_to_default=True, upsample_size=4.0)
        for person in humans:
            H = 0
            check = False
            if person.score < 0.7:continue
            key = person.body_parts.keys()
            MinX,MaxX,MinY,MaxY = 1,0,1,0
            for c in key:
                point = person.body_parts[c]  
                MinX = min(MinX,point.x)
                MinY = min(MinY,point.y)
                MaxX = max(MaxX,point.x)
                MaxY = max(MaxY,point.y)    
            start = (int(MinX*w),int(MinY*h))
            end = (int(MaxX*w),int(MaxY*h))
            if 8 in person.body_parts and 11 in person.body_parts:
                CenterY = int(h*min(person.body_parts[8].y,person.body_parts[11].y))
            elif 8 in person.body_parts:
                CenterY = int(h*person.body_parts[8].y)
            elif 11 in person.body_parts:
                CenterY = int(h*person.body_parts[11].y)
            else:
                continue
            
            if 10 not in person.body_parts or 13 not in person.body_parts:
                H = 1
            
            Sample = []
            if start[1] == end[1] or start[0] == end[0]:continue
            Sample.append(cv2.resize(Ori[start[1]:end[1],start[0]:end[0]],(128,256)))
            Sample.append(cv2.resize(Ori[start[1]:CenterY,start[0]:end[0]],(128,256)))
            x.append(reid.predict(np.array(Sample)))
            box.append([start[1],start[0],end[1],end[0],CenterY])
            ah.append(H)
        ax.append(x)
        aah.append(ah)
        abox.append(box)
        
    if ret==False:break    
    ID = Net.train_all(ax)
    T.append([ID,abox,aah])   
    T = T[0]
    nT = []
    for i in range(len(cap)):nT.append([T[0][i],T[1][i]])
        
    for _ in range(len(cap)):
        out = nT[_]
        result = frames[_]
        if type(out[0]) != type(None):
            for i in range(len(out[0])):
                ID,box = out[0][i],out[1][i]
                x1 = box[1]
                x2 = box[3]
                y1 = box[0]
                y2 = box[2]
                result = cv2.rectangle(result,(x1,y1),(x2,y2),(255,125,0),5)   
                result = cv2.putText(result,str(ID),(box[1],box[2]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)             
        cv2.imshow(str(_),result)
      
    cv2.waitKey(1)
    n+=1
    

            