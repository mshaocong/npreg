# -*- coding: utf-8 -*- 
"""
TODO:
- 把对称性修正
- 能正确处理1对1，1对n的计算

- tensorflow对距离的实现也加入
"""
import numpy as np

def euclidean(xi,x):
    if x.ndim == 1:
        dist = np.sqrt( np.sum((xi-x)**2) ) 
    else:
        dist = np.sqrt( np.sum((xi-x)**2,1) )
    return dist

def sphere(x,y):
    pass

DIST_OPTIONS = {'euclidean': euclidean,'sphere' :sphere}
