# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:33:01 2020

@author: user
"""

import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from GenerateModel import load_model
from Network import network
import pickle
w,h = 360,288
estimator = TfPoseEstimator(get_graph_path('mobilenet_thin'),target_size=(w,h))
reid = load_model('./model/AMSoftmax_0.05.h5')



#Append Video file to here
cap = []
cap.append(cv2.VideoCapture('./VideoDataset/4p-c0.avi'))
cap.append(cv2.VideoCapture('./VideoDataset/4p-c1.avi'))
cap.append(cv2.VideoCapture('./VideoDataset/4p-c2.avi'))
cap.append(cv2.VideoCapture('./VideoDataset/4p-c3.avi'))

# %%
Net = network(1,0.3,40,100,'MDS')
Need_Point = [16,17,2,1,5,8,11,10,13]


clf = []
for i in range(len(cap)):
    file = open('./MDS_pos/'+str(i)+'.pickle','rb')
    clf.append(pickle.load(file))
    file.close()


def predictFoot(box,h):
    if h == 0:
        return np.array([(box[0]+box[2])/2,box[3]])
    else:
        return np.array([box[0]+box[2]/2,box[4]+ 0.8*(box[4]-box[1])])
# %%
n = 1
while True:
    
    X = []
    apos = []
    abox = []
    ahalf = []
    ax = []
    T = []
    frames = []
    for cam in range(len(cap)):

        x = []
        pos = []
        box = []
        half = []
        ret,Ori= cap[cam].read()
            
        if ret==False:break
        frames.append(Ori)
        result = Ori.copy()
        image = Ori.copy()
        humans = estimator.inference(image,resize_to_default=True, upsample_size=4.0)  ##upsample_size = 4.0
        for person in humans:
            H = 0
            check = False
            if person.score < 0.7:continue
            key = person.body_parts.keys()
            if len(key) < 6:continue
            MinX,MaxX,MinY,MaxY = 1,0,1,0
            for c in key:
                point = person.body_parts[c]  
#                result = cv2.circle(result,(int(point.x*1920),int(point.y*1080)),8,(255,0,0),-1)
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
                
            half.append(H)
            
            Sample = []
            if start[1] == end[1] or start[0] == end[0]:continue
            Sample.append(cv2.resize(Ori[start[1]:end[1],start[0]:end[0]],(128,256)))
            Sample.append(cv2.resize(Ori[start[1]:CenterY,start[0]:end[0]],(128,256)))
            predict_pos = clf[cam].predict([predictFoot([start[0],start[1],end[0],end[1],CenterY],H)])[0]
            x.append(reid.predict(np.array(Sample)))
            pos.append(predict_pos)
            box.append([start[1],start[0],end[1],end[0],CenterY,H])  
            
        ax.append(x)
        apos.append(pos)
        abox.append(box)
        ahalf.append(half)

    if ret==False:break
    ID = Net.train_all(ax,apos)
    T.append([ID,apos,abox,ahalf])
     
    T = T[0]
    nT = []
    for i in range(len(cap)):nT.append([T[0][i],T[1][i],T[2][i]])
        
    for _ in range(len(cap)):
        out = nT[_]
        result = frames[_]
        if type(out[0]) != type(None):
            for i in range(len(out[0])):
                ID,pos,box = out[0][i],out[1][i],out[2][i]
                x1 = box[1]
                x2 = box[3]
                y1 = box[0]
                y2 = box[2]
                c = box[4]
                check = box[5]
                result = cv2.rectangle(result,(x1,y1),(x2,y2),(255,125,0),2)   
                if check == 1:result = cv2.rectangle(result,(x1,y1),(x2,c+(c-y1)*check),(125,255,0),2)   
                result = cv2.putText(result,str(ID),(box[1],box[2]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)             
                result = cv2.putText(result,'%.2f %.2f'%(pos[0],pos[1]),(box[1],box[0]+40),cv2.cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(str(_),result)
    cv2.waitKey(1)
    n+=1


    
