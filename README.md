# Nonparametric Regression Models  
Several examples of nonparametric regression models. I will also consider the correlated errors; and I am trying to find some real datasets.

## K-NN-FL
Based on [this paper](https://arxiv.org/abs/1807.11641). Easy to use:
```python
import numpy as np
from fussed_lasso import FusedLasso
X = np.random.rand(20,2) 
y = np.array(range(20))
model = FusedLasso()
model.fit(X,y) 

x = np.array([1.0,1.0]) 
model.predict(x)
```

See 'example.ipynb' for a simulation example.
 
## N-W estimator on manifolds
Based on [this slides](http://www.cs.unc.edu/~lazebnik/fall09/manifold_kernel_regression.pdf). For now, I can do regression on a sphere:
```{python}
model = KernelRegression(dist='sphere')
model.fit(X, y)
```

## Time series with correlated errors
When choosing the best bandwith in kernel smoothing, correlated errors will lead our automated method to select a too small bandwith. A popular way to deal with this situation is Altman's method introduced in *Kernel Smoothing of Data With Correlated Errors*. And I also implement another method from *Kernel Regression in the Presence of Correlated Errors*.
