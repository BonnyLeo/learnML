# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 19:17:07 2022

@author: Bonny B
"""

import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import math

def gradient(x,y):
    n=len(x)
    m_curr = b_curr = 0
    iterations = 1000000
    learning_rate = 0.0002
    prev_cost = 0

    for i in range (iterations):
        y_predict = m_curr * x + b_curr
        cost = 1/n * (sum([val**2 for val in (y-y_predict)]))
        dm = -2/n * sum(x* (y-y_predict))
        db = -2/n * sum((y-y_predict))
        m_curr   = m_curr - (learning_rate * dm)
        b_curr   = b_curr - (learning_rate * db)
        
        if (math.isclose(cost,prev_cost,rel_tol=1e-20)):
            break
        prev_cost = cost
        print ('m {} ,b {} , cost {} ,iter {}'.format(m_curr,b_curr,cost,i))

df = pd.read_csv('E:/work folder/learnML/03_gradient_cost/test_scores.csv')
x = df.math.to_numpy()
y = df.cs.to_numpy()

gradient(x, y)

model = lm.LinearRegression()
model.fit(df[['math']],df.cs)

print ('\n m {} ,b {}  '.format(model.coef_,model.intercept_))