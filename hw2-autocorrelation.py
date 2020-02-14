#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:11:31 2020

@author: songxiaopai
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smfa
import statsmodels.api as sma

df=pd.read_csv('hxq-hw413.csv',sep=',') 
result=smfa.ols(formula='y~x',data=df).fit()
print('Parameters: ', result.params) #const -1.434832 / x 0.176163
#print('e:',result.resid)

#计算残差 e_t 的自相关系数与 DW 统计量
e=result.resid
e1=np.array(e[:-1]) #[e1,...,e19]
e2=np.array(e[1:]) #[e2,...,e20]
#print(e1)
#print(e2)
rho=np.dot(e1,e2)/np.sqrt(np.dot(e1,e1))/np.sqrt(np.dot(e2,e2))
print('Rho:',rho) #0.66
dw=np.sum((e2-e1)**2)/np.sum(e2**2) 
print('DW:',dw) #0<dw=0.667<dl=0.95 postive

#数据进行变量代换，再最小二乘回归
x=df['x']
y=df['y']
rho1=1-dw/2
#print(rho1)
xp=np.array(x[1:])-rho1*np.array(x[:-1])
yp=np.array(y[1:])-rho1*np.array(y[:-1])
#xp=np.array(x[:-1])-rho1*np.array(x[1:])
#yp=np.array(y[:-1])-rho1*np.array(y[1:])
Xp=sma.add_constant(xp)
result2=sma.OLS(yp,Xp).fit()
beta0=result2.params[0]
beta1=result2.params[1]
print('Beta0:',beta0)
print('Beta1:',beta1)
#beta00=result2.params[0]/(1-rho1)

#检查ut的自相关性，计算自相关系数与 DW 统计量
yhatp=beta0+beta1*xp
u=yp-yhatp
u1=np.array(u[:-1])
u2=np.array(u[1:])
rhop=np.dot(u1,u2)/np.sqrt(np.sum(u1**2))/np.sqrt(np.sum(u2**2))
print('Rhop:',rhop) #0.30
#yt=-0.30+0.67y(xt-0.67xt-1)
