# -*- coding: utf-8 -*-

"""
Created on Tue Feb 7 10:42:52 2017

@author: Óscar Barquero Pérez
         Rebeca Goya Esteban
"""

import numpy as np
import matplotlib.pyplot as plt

def mix(n,p):
    """
    Function that creates a mix process of length n and parameter p
    """
    
#Deterministic component
    xx = np.arange(n)
    X = np.sqrt(2)*np.sin((2*np.pi*xx)/12.);
#Stochastic component

    Y = -np.sqrt(3) + 2*np.sqrt(3)*np.random.rand(1,n)
    tt = np.random.rand(1,n)
    Z = tt.copy()
    Z[tt>p] = 0
    Z[tt<= p] = 1

#Final MIX process

    res = (1-Z)*X+Z*Y

    return res.T
    

#test
#N = 200
#p = 0.5

#r = []
#for n in range(100):
#    r.append(mix(N,p).ravel())
    
#rr = np.array(r)





