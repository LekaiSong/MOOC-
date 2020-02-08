#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:11:05 2020

@author: LekaiSong
"""

import statsmodels.api as sm
import pandas as pd

features=['X1','X2','X3','X4']
labels=['Y']
df=pd.read_csv('case1.txt',sep='\t',header=None)
df.columns=labels+features
#df.columns=['Y','X1','X2','X3','X4']
#print(df)
#https://www.jianshu.com/p/ab0c62ee002b

y=df[labels]
#print(y)
X=sm.add_constant(df[features])
#print(X)
model = sm.OLS(y, X)
result = model.fit()
print('Parameters: ', result.params) #1.A
print(result.summary()) #2.B / 3.B / 4.C

df2=df[features].append([{'X1':14,'X2':8.19,'X3':0.27,'X4':104}],ignore_index=True)
#https://blog.csdn.net/weixin_39750084/article/details/81429037
#print(df2)
new_x=sm.add_constant(df2)
prediction=result.predict(new_x[-1:]) #select last row values
print(prediction) #5.A