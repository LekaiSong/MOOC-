#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:27:18 2020

@author: Lekai Song
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def subsets(nums):
    """
    :function: all subsets
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    output = [[]]
    for i in range(len(nums)):
        for j in range(len(output)):
            output.append(output[j]+[nums[i]])
    return output
#https://blog.csdn.net/xiaoxiaoley/article/details/78720378
    
sets=[]
for i in range(1,5):
    sets.append(i)
#print(subsets(sets)[1:-1])
        
df=pd.read_csv('hald.csv')
rxy=df.corr(method='pearson')
print('rxy:',rxy) #rmax=p(x2,x4)=-0.9730

dfx=list(df)
dfx.remove('y')
dfx=df[dfx]
dfy=df.y
rxx=dfx.corr(method='pearson')
#print(rxx)
C=np.linalg.inv(rxx)
#print(C)
VIF=np.diag(C).round(2)
print('VIF:',VIF) #38.5 254.42 46.87 282.51

df_scaled=(df-df.mean())/df.std()
A_scaled=np.array(df_scaled)
#print(A_scaled) #ndarray,not dataframe
x1x2=A_scaled[:,[1,2]]
x3x4=A_scaled[:,[3,4]]
A=np.array(df)
X=A[:,1:]
#print(X)
B=np.dot(X.T,X)
ev,evct=np.linalg.eig(B)
kk=ev.max()/ev.min()
print('lambda1/lambda2:',kk) #423.7

lr1=OLS(dfy, add_constant(x1x2)).fit()
lr2=OLS(dfy, add_constant(x3x4)).fit()
print('AIC:',lr1.aic,lr2.aic) #x1x2=62.31 x3x4=76.74 x2x4=97.51

xmin=A_scaled[:,:]
nmin=sets
lrmin=OLS(dfy,add_constant(xmin)).fit()
for n in subsets(sets)[1:-1]:
    xx=A_scaled[:,n]
    lr=OLS(dfy, add_constant(xx)).fit()
#    print(lr.aic)
    if lr.aic<lrmin.aic: 
        lrmin.aic=lr.aic
        nmin=n
print('AICmin:',lrmin.aic,'Combination:',nmin) #x1,x2,x3,x4