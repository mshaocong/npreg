import numpy as np 
import tensorflow as tf 

from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient

from kernel import KERNEL_OPTIONS
from metric import DIST_OPTIONS

MANIFOLD_OPTIONS = {'sphere': Sphere(3)}
 
class KernelRegression:
    def __init__(self, dist="euclidian", kernel="gaussian", bws=1):
        #initializing
        if dist in DIST_OPTIONS:
            self.dist_name = dist
            self.dist = DIST_OPTIONS[dist]
        else:
            self.dist_name = "undefined"
            self.dist = dist
            
        if kernel in KERNEL_OPTIONS:
            self.kernel = KERNEL_OPTIONS[kernel]
        else:
            self.kernel = kernel
            
        self.bws = bws
        self.solver = ConjugateGradient() 
            
    def fit(self, X, y):
        self.__X = X
        self.__y = y
    
    #只预测一个点
    def __predict(self, x):
        kernel = self.kernel
        
        #define cost function
        q = tf.Variable( [1,0,0] , dtype="float32")
        v = kernel(self.__X - x, self.bws).astype("float32")
        
        cost =  tf.tensordot( v, tf.square( self.dist(  self.__y, q)), 1)
        #solve the opt problem
        if self.dist_name in MANIFOLD_OPTIONS:
            manifold = MANIFOLD_OPTIONS[self.dist_name]
            problem = Problem(manifold=manifold, cost=cost, arg=q, verbosity=0)
        else:
            raise
        solver = self.solver
        Xopt = solver.solve(problem)
        return(Xopt)
        
    #预测所有的点
    def predict(self, X):
        try:
            attempt = iter(X)
            predicted_value = []
            for x in X:
                predicted_value.append( self.__predict(x) )
            return np.array(predicted_value)
        except TypeError as te:
            return self.__predict(X) 
    
    def fitted(self):
        predicted_value = []
        for x in self.__X:
            predicted_value.append( self.__predict(x) )
        return np.array(predicted_value)
    
    #def score(self):
    #    pass
    
    def set_dist(self, dist):
        self.dist = dist
    
    def set_kernel(self, kernel):
        self.kernel = kernel
    
    def set_bws(self, bws):
        self.bws = bws
    
    def set_solver(self, solver):
        self.solver = solver