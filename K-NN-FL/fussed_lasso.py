# -*- coding: utf-8 -*- 
from graph import kneighbors_graph
from metric import DIST_OPTIONS

from pymanopt.manifolds import Euclidean 
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient

import numpy as np
import tensorflow as tf

"""
FusedLasso(): K-NN-FL model 
Parameters:
    k_neighbors: =5 by default. 
    metric: ='euclidean' by default.
Method:
    fit(X, y, alpha)
    predict(X)
"""
class FusedLasso(): 
    def __init__(self, k_neighbors=5, metric='euclidean'):  
        self.k_neighbors = k_neighbors
        if metric in DIST_OPTIONS:
            self.metric = DIST_OPTIONS[metric]
        else:
            self.metric = metric
        
    def fit(self, X, y, alpha=1, verbosity=0):
        self.__X = X
        self.__y = y
        
        #generate knn graph
        graph = kneighbors_graph(X, self.k_neighbors, self.metric)
        self.graph = graph[0]
        self.incident_matrix = graph[1]
        
        
        #get theta
        G = graph[1]
        n = len(y)
        tf.reset_default_graph()
        theta = tf.get_variable('theta',shape=[1,n], dtype=tf.float32)
        Y=y.astype('float32') 
        
        mse = tf.reduce_sum(tf.squared_difference(Y , theta))
        penalty =  alpha*tf.norm(tf.matmul(G, theta,transpose_b=True),ord=1)
        
        cost = mse + penalty
        optimizer = tf.train.AdamOptimizer(0.01)
        train = optimizer.minimize(cost) 
        
        cost_record = []
        
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                each_record = {'mse':[],'penalty':[] } 
                each_record['mse'] = sess.run( mse )
                each_record['penalty'] = sess.run( penalty ) 
                cost_record.append(each_record)
                sess.run(train)
            self.theta = sess.run(theta).flatten() 
        self.cost_record = cost_record
        
    def __k(self, i, x):
        #找到距离x最近的k个点，判断x_i是否在其中；在就返回1，不在就返回0
        distance_to_x = self.metric(x, self.__X )
        k_nearest_points = np.argsort(distance_to_x)[0:self.k_neighbors] #get the index of the smallest k elements 
        if i in k_nearest_points:
            return 1
        else:
            return 0
        
    def __predict(self, x):  
        theta = self.theta
        tmp = self.__X
        num_obs = tmp.shape[0]
        
        d = 0.0
        s = 0.0
        for i in range(num_obs): 
            k = self.__k(i, x)
            d += k
            s += k*theta[i] 
        return s/d
    
    def predict(self, X):
        if X.ndim == 1:
            return self.__predict(X)
        else:
            estimate = []
            for x in X:
                estimate.append(self.__predict(x))
            return np.array(estimate)