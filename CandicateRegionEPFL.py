# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:09:23 2020

@author: user
"""

import numpy as np


Dict = np.load('./IOU_POS/epfl_lab_X.npy')
Y = np.load('./IOU_POS/epfl_lab_y.npy')


def IOU(rec1, rec2):
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    sum_area = S_rec1 + S_rec2
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)  


def candidate(cam,box):
    Dc = Dict[:,cam]
    Max = 0
    center = (0,0)
    for i in range(len(Dict)):
            iou = IOU(Dc[i],box)
            if Max <= iou:
                Max = iou
                center = Y[i]
    if Max < 0.2:center=(3,3)
    return np.array(center,dtype=np.int)


def getIOU(rect,cam,pos):
    return IOU(rect,Dict[cam][pos])
    
    
                
    