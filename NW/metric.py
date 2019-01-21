#Distance - 这个部分依赖q。所以需要计算梯度。所以要用tf 
import tensorflow as tf

def euclidean(x,y):
    pass
    #return tf.norm(x, y, ord='euclidean') 

#note: it is unit sphere
def sphere(x, y):
    #x,y must be 3-dim
    inner = tf.tensordot(x,y,axes=1)
    return tf.acos(inner)

DIST_OPTIONS = {'euclidian': euclidean,'sphere' :sphere} 