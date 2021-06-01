import numpy as np
import tensorflow as tf
from utils import build_ker
from settings import tf_dtype as tf_dtype
from settings import np_dtype as np_dtype
from settings import jitter
from utils import vec_to_tri, select_Z

class GP_activ:
    def __init__(self,g,l,M,D,ker,init_af,Z_run):
        self.mf, self.ker, self.D, self.M = tf.zeros_like, ker, D, M
        Z_init = select_Z(Z_run,"linspace",M)
        log_l_init = D*[tf.log(l)]
        log_g_init = D*[tf.log(g)]
        self.log_l = tf.Variable(tf.cast(log_l_init,dtype=tf_dtype),name="log_l")           # D
        self.log_g = tf.Variable(tf.cast(log_g_init,dtype=tf_dtype),name="log_g")           # D

        self.Z = tf.Variable(tf.cast(Z_init,tf_dtype),name="Z")                             # D,M
        self.Kzz = build_ker(self.Z,self.Z,self.log_g,self.log_l,self.ker)                  # D,M,M
        self.Lzz = tf.cholesky(self.Kzz + jitter*tf.eye(self.M,dtype=tf_dtype)[None,:,:])     # D,M,M

        q_mu_init = tf.matrix_triangular_solve(self.Lzz,(init_af(self.Z)-self.mf(self.Z))[:,:,None])[:,:,0]
        self.q_sqrt_slt = tf.Variable(np.broadcast_to(1e-5*np.eye(self.M,dtype=np_dtype)[np.tril_indices(self.M)],
                                                      [self.D,self.M*(self.M+1)//2]),name="q_sqrt_slt")
        self.q_sqrt = vec_to_tri(self.q_sqrt_slt,self.M)   # D,M,M
        self.q_mu = tf.Variable(q_mu_init,name="q_mu")   # D,M

    def sample_from_conditional(self, X):
        S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
        X_flat = tf.reshape(X, [S * N, D])
        mean_flat, var_flat = self.conditional(X_flat)
        mean,var = [tf.reshape(m, [S, N, D]) for m in [mean_flat, var_flat]]

        z = tf.random_normal(tf.shape(mean),dtype=tf_dtype)
        samples = mean+z*(var**0.5)

        return samples, mean, var

    def conditional(self,X_new):  # X_new: N,D
        Kzx = build_ker(self.Z,tf.transpose(X_new),self.log_g,self.log_l,self.ker) # D,M,N
        A = tf.matrix_triangular_solve(self.Lzz,Kzx,lower=True) # D,M,N
        mean = self.mf(X_new)+tf.einsum('ijk,ij->ki',A,self.q_mu)    # N,D

        SK = -tf.eye(self.M,dtype=tf_dtype)[None,:,:]+tf.matmul(self.q_sqrt,self.q_sqrt,transpose_b=True) # D,M,M
        B = tf.matmul(SK,A)    # D,M,N
        delta_cov = tf.transpose(tf.reduce_sum(A*B,1))  # N,D
        Kxx_diag = tf.exp(self.log_g)[None,:] # N,D
        var =  Kxx_diag + delta_cov       # N,D

        return mean, var

    def KL(self):
        KL = -0.5 * self.M*self.D
        KL -= tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(self.q_sqrt))))
        KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
        KL += 0.5 * tf.reduce_sum(self.q_mu**2)
        return KL
