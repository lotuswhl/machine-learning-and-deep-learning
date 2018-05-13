# -*- coding: utf-8 -*-
import numpy as np
import math
def matrix_factorization(Original,P,Q,K,steps=5000,lr=0.0002,regl=0.02):
    Q=Q.T
    for step in range(steps):
        for i in range(len(P)):
            for j in range(len(R[i])):
                # 只更新评分大于0的
                if Original[i,j]>0:
                    eij = Original[i,j]-np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i,k]=P[i,k]+lr*(2*eij*Q[k,j]-regl*P[i,k])
                        Q[k,j]=Q[k,j]+lr*(2*eij*P[i,k]-regl*Q[k,j])
        err = 0.0
        for i in range(len(P)):
            for j in range(len(R[i])):
                if Original[i][j]>0:
                    err=err+math.pow(np.dot(P[i,:],Q[:,j])-Original[i][j],2)
                    for k in range(K):
                        err=err+(regl/2)*(math.pow(P[i][k],2)+math.pow(Q[k][j],2))
        if err < 1e-3:
            break
    return (P,Q.T)


R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
R=np.array(R)
U=len(R)
I=len(R[0])
K=2
P=np.random.rand(U,K)
Q=np.random.rand(I,K)

newP,newQ=matrix_factorization(R,P,Q,K)
newR=np.dot(newP,newQ.T)

print(newR)

