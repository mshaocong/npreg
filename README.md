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

See 'example.ipynb' for Example 2.2 in that paper.  

<Explain how it works here.>

## N-W estimator on manifolds
Only 1-dim case.  

## Frechet regression
TODO.
