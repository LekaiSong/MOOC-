#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:40:22 2020

@author: Lekai Song
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV

df=pd.read_csv('unemp.csv')
dfx_title=list(df)
dfx_title.remove('y')
dfx=df[dfx_title]
dfy=df.y
dfy_scaled=(dfy-dfy.mean())/dfy.std()
#print(type(dfy_scaled))
print("Data Loading")
print("---------------------------------------------------------------------")

#岭回归，参数估计，固定岭参数
X=dfx.iloc[:,0:5]
#print(X)
y=dfy_scaled
reg01=Ridge(alpha=0.15).fit(X,y)
print('Ridge(alpha=0.15) score:',reg01.score(X,y).round(5)) #0.98513
print('Ridge(alpha=0.15) coefficients:',reg01.coef_.round(5),'\n') #[-0.05087 0.54623 0.39501 -0.12857 -0.03614]
print("---------------------------------------------------------------------")

#岭回归，按 CV 标准自动选择岭参数
alphas=np.linspace(0.0001,0.5,1000)
reg02=RidgeCV(alphas).fit(X,y)
print('RidgeCV score:',reg02.score(X,y).round(5))
print('RidgeCV alpha:',reg02.alpha_.round(5)) #0.33737
print('RidgeCV coefficients:',reg02.coef_.round(5),'\n')
print("---------------------------------------------------------------------")

#lasso求解
count=0
lamb=0.05
lasso_reg=Lasso(alpha=lamb)
lasso_reg.fit(dfx,dfy)
print('Lasso Intercept:',lasso_reg.intercept_)
print('Lasso Coef:','\n',lasso_reg.coef_)
#print(type(lasso_reg.coef_))
for n in lasso_reg.coef_:
    if((n>1e-5)or(n<-1e-5)):
        count=count+1
#        print(n)
print('# of |Coef|>1e-5:',count,'\n') #5
print("---------------------------------------------------------------------")
      
#lasso超参数选择
lasso_reg2=LassoCV(cv=20).fit(dfx,dfy)
print('LassoCV Incercept:',lasso_reg2.intercept_)
print('LassoCV Coef:','\n',lasso_reg2.coef_)
print('LassoCV Alpha:',lasso_reg2.alpha_,'\n') #0.03393
print("---------------------------------------------------------------------")

#改用弹性网络方法
count2=0
lamb1=0.15
ElasticNet_reg=ElasticNet(alpha=lamb1,l1_ratio=0.95).fit(dfx,dfy)
print('ElasticNet Incercept:',ElasticNet_reg.intercept_)
print('ElasticNet Coef:','\n',ElasticNet_reg.coef_.T)
for i in ElasticNet_reg.coef_.T:
    if((i>1e-5)or(i<-1e-5)):
        count2=count2+1
#        print(n)
print('# of |Coef|>1e-5:',count2,'\n') #8
print("---------------------------------------------------------------------")
