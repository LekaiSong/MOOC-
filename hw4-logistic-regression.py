#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:58:07 2020

@author: Lekai Song
"""
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df=pd.read_csv('shuttle_train_binary.csv')
df2=pd.read_csv('shuttle_test_binary.csv')
df3=pd.read_csv('shuttle_train_ternary.csv')
features=['x1','x2','x3','x4','x5','x6','x7','x8','x9']
labels=['y']

model=LogisticRegression().fit(df[features],df[labels].values.ravel())
#https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
print('Default Coef:',model.coef_)
print('First Two Coefs:',model.coef_[0][0],'and',model.coef_[0][1]) #16,-5
print('Default Intercept:',model.intercept_)

prob=model.predict_proba(df[features][-1:])
print('Prob:',prob) #0.956,0.04

#计算混淆矩阵
pred=model.predict(df2[features])
confusion=metrics.confusion_matrix(df2[labels],pred)
print(confusion)
plt.matshow(confusion)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('Predict')
plt.xlabel('Actual')
#plt.show()
#https://blog.csdn.net/kane7csdn/article/details/83756583?utm_source=distribute.pc_relevant.none-task

#Suppose for y=1
TP=confusion[1][1]
FN=confusion[1][0]
recall=TP/(TP+FN)
print('Recall:',recall) #0.869

fpr,tpr,thresholds=metrics.roc_curve(df2[labels],pred)
auc=metrics.auc(fpr,tpr)
print('AUC:',auc) #0.928

model2=LogisticRegression(multi_class='multinomial',solver='sag').fit(df3[features],df3[labels].values.ravel())
pred2=model2.predict(df3[features])
confusion2=metrics.confusion_matrix(df3[labels],pred2)
print(confusion2)
print('# Of y1 Disclassified To Else:',confusion2[0][1]+confusion2[0][2]) #478