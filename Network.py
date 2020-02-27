# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:44:06 2020

@author: user
"""
import numpy as np

def E(a,b):
    return np.linalg.norm(a-b) 

class network(object):
    def __init__(self,th,pth,MaxNum,MaxCheck,pos_loc):
        self.th = th
        self.AllCentroid = []
        self.AllCluster = []
        self.HalfCentroid = []
        self.HalfCluster = []
        self.pos = []
        self.check = []	
        self.Candidate = []
        self.AllClusterT = []
        self.HalfClusterT = []
        self.MaxNum = MaxNum
        self.MaxCheck = MaxCheck
        self.pth = pth
        self.pos_loc = pos_loc #IOU or MDS
        
    def train_all(self,X,pos):
        Weight = [4,3,2,1]
        out = []
        for i in range(len(X)):
            out.append(self.train_batch(X[i],pos[i]))
        for i in range(len(self.Candidate)):self.Candidate[i]-=(1/Weight[len(X)-1])
        for e in range(len(self.check)):self.check[e]+=(1/Weight[len(X)-1])
        return out
    
    def linkage(self,X,pos):
        w = []
        B = [-0.3,-0.2,-0.1,-0.05]
        for i in range(len(self.AllCluster)):
            P = E(pos,self.pos[i])
            AllE = E(X[0],self.AllCentroid[i])
            HalfE = E(X[1],self.HalfCentroid[i])
            tmp = min(AllE,HalfE) 
            print(P)
            if self.check[i] < self.MaxCheck:
                if P < self.pth:
                    if self.pos_loc == 'IOU':
                        tmp = tmp + B[int(P)]
                    elif self.pos_loc == 'MDS':
                        tmp = tmp + P
                else:
                    tmp = 10
            else:
                tmp = 10
            w.append(tmp)
        return w

    def Pair(self,M):
        out = []
        check = []
        while M.shape[0] * M.shape[1] * 10 > np.sum(M):
            e = np.unravel_index(np.argmin(M),M.shape)
            out.append([e[0],e[1]])
            M[e[0],:] = 10
            M[:,e[1]] = 10
            check.append(e[1])
        check = set(range(M.shape[1])) - set(check)
        for e in check:out.append([-1,e])
        return out
    
    def UpdateCluster(self,Cn,X):
        if len(self.AllCluster) > self.MaxNum:
            self.AllCluster[Cn].pop(0)
            self.AllCluster[Cn].append(X[0])
        else:
            self.AllCluster[Cn].append(X[0])
            
        if len(self.HalfCluster) > self.MaxNum:
            self.HalfCluster[Cn].pop(0)
            self.HalfCluster[Cn].append(X[1])
        else:
            self.HalfCluster[Cn].append(X[1])
        self.AllCentroid[Cn] = np.mean(np.array(self.AllCluster[Cn]),axis = 0)
        self.HalfCentroid[Cn] = np.mean(np.array(self.HalfCluster[Cn]),axis = 0)
    
    def NewPerson(self,X,pos):
        MIN = 10
        INDEX = -1
        out = None
        for _ in range(len(self.AllClusterT)):
            All = E(X[0],np.mean(np.array(self.AllClusterT[_]),axis=0))
            Half = E(X[1],np.mean(np.array(self.HalfClusterT[_]),axis=0))
            X2Tmp = min(All,Half)
            if X2Tmp < self.th and X2Tmp < MIN:
                MIN = X2Tmp
                INDEX = _
        if INDEX != -1:
            self.AllClusterT[INDEX].append(X[0])
            self.HalfClusterT[INDEX].append(X[1])
            for i in range(len(self.AllClusterT[INDEX])-40):self.AllClusterT[INDEX].pop(0)
            for i in range(len(self.HalfClusterT[INDEX])-40):self.HalfClusterT[INDEX].pop(0)
            self.Candidate[INDEX]+=1
            if self.Candidate[INDEX] > 20:
                self.Candidate[INDEX] = -1
                self.AllCluster.append(self.AllClusterT[INDEX])
                self.HalfCluster.append(self.HalfClusterT[INDEX])
                self.AllCentroid.append(np.mean(np.array(self.AllClusterT[INDEX]),axis=0))
                self.HalfCentroid.append(np.mean(np.array(self.HalfClusterT[INDEX]),axis=0))
                self.check.append(0)
                self.pos.append(pos)
                out = len(self.pos) - 1
            else:
                out = None
        else:
            out = None
            self.Candidate.append(1)
            self.AllClusterT.append([X[0]])
            self.HalfClusterT.append([X[1]])
        return out
    
    def train_batch(self,X,pos):
        if len(X) == 0:return None
        x2C = []
        C = list(range(len(X)))
        for x in range(len(X)):
            W = self.linkage(X[x],pos[x])
            for i in range(len(W)):x2C.append(W[i])
        
        x2C = np.array(x2C).reshape((-1,len(X)),order='F')
        pair = self.Pair(x2C.copy())
        for p in pair:
            if p[0] != -1:
                if x2C[p[0],p[1]] < self.th:
                    self.UpdateCluster(p[0],X[p[1]])
                    self.pos[p[0]] = pos[p[1]]
                    self.check[p[0]] = 0
                    C[p[1]] = p[0]
                else:
                    C[p[1]] = self.NewPerson(X[p[1]],pos[p[1]])
            else:
                if len(x2C) > 0:
                    C[p[1]] = self.NewPerson(X[p[1]],pos[p[1]])
                else:
                    self.AllCluster.append([X[p[1]][0]])
                    self.HalfCluster.append([X[p[1]][1]])
                    self.AllCentroid.append(X[p[1]][0])
                    self.HalfCentroid.append(X[p[1]][1])
                    self.pos.append(pos[p[1]])
                    self.check.append(0)
                    C[p[1]] = len(self.pos) - 1
        print(x2C)            
        tmp = [i for i in range(len(self.Candidate)) if self.Candidate[i] < 0]
        tmp = set(list(range(len(self.Candidate)))) - set(tmp)
        Candidate = [self.Candidate[_] for _ in tmp]
        AllClusterT = [self.AllClusterT[_] for _ in tmp]
        HalfClusterT = [self.HalfClusterT[_] for _ in tmp]
        self.Candidate = Candidate
        self.AllClusterT = AllClusterT       
        self.HalfClusterT = HalfClusterT  
        return C

class network_nopos(object):
    def __init__(self,th,MaxNum,MaxCheck):
        self.th = th
        self.AllCentroid = []
        self.AllCluster = []
        self.HalfCentroid = []
        self.HalfCluster = []
        self.pos = []
        self.check = []	
        self.Candidate = []
        self.AllClusterT = []
        self.HalfClusterT = []
        self.MaxNum = MaxNum
        self.MaxCheck = MaxCheck

        
    def train_all(self,X):
        Weight = [4,3,2,1]
        out = []
        for i in range(len(X)):
            out.append(self.train_batch(X[i]))
        for i in range(len(self.Candidate)):self.Candidate[i]-=(1/Weight[len(X)-1])
        for e in range(len(self.check)):self.check[e]+=(1/Weight[len(X)-1])
        return out
    
    def linkage(self,X):
        w = []
        for i in range(len(self.AllCluster)):

            AllE = E(X[0],self.AllCentroid[i])
            HalfE = E(X[1],self.HalfCentroid[i])
            tmp = min(AllE,HalfE) 
            if self.check[i] > self.MaxCheck:tmp = 10
            w.append(tmp)
        return w

    def Pair(self,M):
        out = []
        check = []
        while M.shape[0] * M.shape[1] * 10 > np.sum(M):
            e = np.unravel_index(np.argmin(M),M.shape)
            out.append([e[0],e[1]])
            M[e[0],:] = 10
            M[:,e[1]] = 10
            check.append(e[1])
        check = set(range(M.shape[1])) - set(check)
        for e in check:out.append([-1,e])
        return out
    
    def UpdateCluster(self,Cn,X):
        if len(self.AllCluster) > self.MaxNum:
            self.AllCluster[Cn].pop(0)
            self.AllCluster[Cn].append(X[0])
        else:
            self.AllCluster[Cn].append(X[0])
            
        if len(self.HalfCluster) > self.MaxNum:
            self.HalfCluster[Cn].pop(0)
            self.HalfCluster[Cn].append(X[1])
        else:
            self.HalfCluster[Cn].append(X[1])
        self.AllCentroid[Cn] = np.mean(np.array(self.AllCluster[Cn]),axis = 0)
        self.HalfCentroid[Cn] = np.mean(np.array(self.HalfCluster[Cn]),axis = 0)
    
    def NewPerson(self,X):
        MIN = 10
        INDEX = -1
        out = None
        for _ in range(len(self.AllClusterT)):
            All = E(X[0],np.mean(np.array(self.AllClusterT[_]),axis=0))
            Half = E(X[1],np.mean(np.array(self.HalfClusterT[_]),axis=0))
            X2Tmp = min(All,Half)
            if X2Tmp < self.th and X2Tmp < MIN:
                MIN = X2Tmp
                INDEX = _
        if INDEX != -1:
            self.AllClusterT[INDEX].append(X[0])
            self.HalfClusterT[INDEX].append(X[1])
            for i in range(len(self.AllClusterT[INDEX])-40):self.AllClusterT[INDEX].pop(0)
            for i in range(len(self.HalfClusterT[INDEX])-40):self.HalfClusterT[INDEX].pop(0)
            self.Candidate[INDEX]+=1
            if self.Candidate[INDEX] > 20:
                self.Candidate[INDEX] = -1
                self.AllCluster.append(self.AllClusterT[INDEX])
                self.HalfCluster.append(self.HalfClusterT[INDEX])
                self.AllCentroid.append(np.mean(np.array(self.AllClusterT[INDEX]),axis=0))
                self.HalfCentroid.append(np.mean(np.array(self.HalfClusterT[INDEX]),axis=0))
                self.check.append(0)
                out = len(self.check) - 1
            else:
                out = None
        else:
            out = None
            self.Candidate.append(1)
            self.AllClusterT.append([X[0]])
            self.HalfClusterT.append([X[1]])
        return out
    
    def train_batch(self,X):
        if len(X) == 0:return None
        x2C = []
        C = list(range(len(X)))
        for x in range(len(X)):
            W = self.linkage(X[x])
            for i in range(len(W)):x2C.append(W[i])
        
        x2C = np.array(x2C).reshape((-1,len(X)),order='F')
        print(x2C)
        pair = self.Pair(x2C.copy())
        for p in pair:
            if p[0] != -1:
                if x2C[p[0],p[1]] < self.th:
                    self.UpdateCluster(p[0],X[p[1]])
                    self.check[p[0]] = 0
                    C[p[1]] = p[0]
                else:
                    C[p[1]] = self.NewPerson(X[p[1]])
            else:
                if len(x2C) > 0:
                    C[p[1]] = self.NewPerson(X[p[1]])
                else:
                    self.AllCluster.append([X[p[1]][0]])
                    self.HalfCluster.append([X[p[1]][1]])
                    self.AllCentroid.append(X[p[1]][0])
                    self.HalfCentroid.append(X[p[1]][1])
                    self.check.append(0)
                    C[p[1]] = len(self.check) - 1
                    
        tmp = [i for i in range(len(self.Candidate)) if self.Candidate[i] < 0]
        tmp = set(list(range(len(self.Candidate)))) - set(tmp)
        Candidate = [self.Candidate[_] for _ in tmp]
        AllClusterT = [self.AllClusterT[_] for _ in tmp]
        HalfClusterT = [self.HalfClusterT[_] for _ in tmp]
        self.Candidate = Candidate
        self.AllClusterT = AllClusterT       
        self.HalfClusterT = HalfClusterT  
        return C