#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:04:33 2020

@author: Lekai Song
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

data=pd.read_csv('case3.csv')
x=data['x']
y=data['y']

M1=np.vstack((np.ones_like(x),x)).T 
M2=np.vstack((np.ones_like(x),x,x**2)).T
M3=np.vstack((np.ones_like(x),x,x**2,x**3)).T

res1=sm.OLS(y,M1).fit()
res2=sm.OLS(y,M2).fit()
res3=sm.OLS(y,M3).fit()

for i in range(1,4):
    exec('print(res{}.summary(), end=" ")'.format(i)) #AIC=112/45/47 coef2=0.0964 beta1=0.001 beta11=0.000
#    https://www.cnblogs.com/technologylife/p/9211324.html
    
y2_fitted=res2.fittedvalues
print(y2_fitted) #v=8 y=10.246