import tensorflow as tf
import numpy as np
from settings import tf_dtype as tf_dtype
from settings import np_dtype as np_dtype

def build_ker(X,Y,log_g,log_l,ker):
    r = tf.abs(X[:,:,None]-Y[:,None,:])/tf.exp(log_l)[:,None,None] # D,N_X,N_Y
    if ker=='RBF':
        res = tf.exp(log_g)[:,None,None]*tf.exp(-r**2/2)
    elif ker=='TRI':
        res = tf.exp(log_g)[:,None,None]*tf.maximum(tf.cast(0.0,tf_dtype),1.0-r)
    return res                     # D,N_X,N_Y

# Credit to GPflow
def vec_to_tri(vectors, N):
    """
    Takes a D x M tensor `vectors' and maps it to a D x matrix_size X matrix_sizetensor
    where the where the lower triangle of each matrix_size x matrix_size matrix is
    constructed by unpacking each M-vector.
    Native TensorFlow version of Custom Op by Mark van der Wilk.
    def int_shape(x):
        return list(map(int, x.get_shape()))
    D, M = int_shape(vectors)
    N = int( np.floor( 0.5 * np.sqrt( M * 8. + 1. ) - 0.5 ) )
    # Check M is a valid triangle number
    assert((matrix * (N + 1)) == (2 * M))
    """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)

def select_Z(Z_run,method,M):
    Z_run = tf.transpose(Z_run) # D,N
    if method=="linspace":
        mm = tf.reduce_min(Z_run,axis=1)
        MM = tf.reduce_max(Z_run,axis=1)
        return(tf.map_fn(lambda x:tf.linspace(x[0]-0.1,x[1]+0.1,M),(mm,MM),dtype=tf_dtype))

def glorot_unif_samp(shape):
    lim = np.sqrt(6.0/(shape[0]+shape[1]))
    return np.random.uniform(low=-lim,high=lim,size=shape)
