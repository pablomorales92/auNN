import numpy as np
import tensorflow as tf
from GP_activ import GP_activ
from settings import tf_dtype as tf_dtype
from settings import np_dtype as np_dtype
from utils import glorot_unif_samp

class auNN:
    def __init__(self, X, y, arch, M, ker="TRI",init_prior_gamma=1.0, init_prior_l=1.0):
        self.X,self.N,self.y,self.arch,self.M,self.ker = X,X.shape[0],y,arch,M,ker
        self.init_af = tf.nn.relu if ker=='TRI' else tf.zeros_like
        self.init_prior_gamma,self.init_prior_l = init_prior_gamma,init_prior_l

        self.m_X = np.mean(self.X,0)
        self.std_X = np.std(self.X,0)
        self.m_y = np.mean(self.y,0)
        self.std_y = np.std(self.y,0)
        self.X_nor = np.zeros(self.X.shape)
        self.X_nor[:,self.std_X!=0] = (self.X[:,self.std_X!=0]-self.m_X[self.std_X!=0])/self.std_X[self.std_X!=0]
        self.y_nor = (self.y-self.m_y)/self.std_y

        self.nELBO, self.prediction_m, self.prediction_std, self.test_loglik = self.build_graph()

        self.lr = tf.placeholder(tf_dtype,shape=[],name="lr")
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.nELBO,name="train_op")
        self.trained_epochs = 0

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)


    def build_graph(self):

        arch_extended = [self.X.shape[1]]+self.arch      # [D_0,D_1,...,D_L]
        self.X_plh = tf.placeholder(tf_dtype,[None, arch_extended[0]], name='features')      # N_mb,D_0
        self.y_plh = tf.placeholder(tf_dtype,[None, self.y.shape[1]], name='outputs')        # N_mb,1
        self.N_mb = tf.shape(self.X_plh)[0]
        self.num_samples = tf.placeholder(tf_dtype, name='num_samples')

        self.activs = []
        self.F_samples = [tf.tile(self.X_plh[None,:,:],[self.num_samples,1,1])]         # [S,N_mb,D^1;...;S,N_mb,D^L]
        self.F_mean = []                                                                # [S,N_mb,D^1;...;S,N_mb,D^L]
        self.F_var = []                                                                 # [S,N_mb,D^1;...;S,N_mb,D^L]
        self.A = []                                                                  # [S,N_mb,D^1;...;S,N_mb,D^L]
        self.W = []                                                                     # [D^0,D^1;...;D^{L-1},D^L]
        self.b = []                                                                     # [D^1;...;D^L]
        KL_term = []
        Z_run = tf.constant(self.X_nor) # N,D^l
        for l in range(len(self.arch)):
            with tf.variable_scope("layer_{}".format(l+1)):
                self.W.append(tf.Variable(glorot_unif_samp([arch_extended[l],arch_extended[l+1]]),dtype=tf_dtype,name="W"))
                self.b.append(tf.Variable(np.zeros(self.arch[l]),dtype=tf_dtype,name="b"))
                Z_run = tf.einsum('jk,kl->jl',Z_run,self.W[-1].initialized_value())+self.b[-1].initialized_value()[None,:]
                self.activs.append(GP_activ(g=self.init_prior_gamma,l=self.init_prior_l,
                                              M=self.M[l],D=self.arch[l],ker=self.ker,
                                              init_af=self.init_af,Z_run=Z_run))
            Z_run = self.init_af(Z_run) # N,D^{l+1}
            self.A.append(tf.einsum('ijk,kl->ijl',self.F_samples[-1],self.W[-1])+self.b[-1][None,None,:])
            F_samples,F_mean,F_var = self.activs[-1].sample_from_conditional(self.A[-1])
            self.F_samples.append(F_samples)
            self.F_mean.append(F_mean)
            self.F_var.append(F_var)
            KL_term.append(self.activs[-1].KL())
        self.F_samples.pop(0)

        self.log_noise_std = tf.Variable(tf.cast(tf.log(0.1),tf_dtype),name="log_noise_std")

        sq_diff = (self.y_plh-self.F_mean[-1])**2   # S,N_mb,1
        self.data_term = (-0.5*tf.cast(self.N_mb,tf_dtype)*tf.cast(tf.log(2*np.pi),tf_dtype)-tf.cast(self.N_mb,tf_dtype)*self.log_noise_std+
                          -0.5*tf.reduce_mean(tf.reduce_sum(sq_diff+self.F_var[-1],axis=1))/tf.exp(2*self.log_noise_std))
        self.KL_term = tf.reduce_sum(KL_term)
        nELBO = -self.data_term + self.KL_term*(tf.cast(self.N_mb,tf_dtype)/self.N)
        prediction_m = tf.reduce_mean(self.F_mean[-1],axis=0,name="prediction_m")     # N_mb,1
        prediction_std = tf.sqrt(tf.reduce_mean(self.F_mean[-1]**2+self.F_var[-1]+tf.exp(2*self.log_noise_std),axis=0)-prediction_m**2,name="prediction_std")  # N_mb,1
        test_loglik = tf.reduce_logsumexp(-0.5*tf.cast(tf.log(2*np.pi),tf_dtype)-0.5*tf.log(tf.exp(2*self.log_noise_std)+self.F_var[-1])
                                          -0.5*sq_diff/(tf.exp(2*self.log_noise_std)+self.F_var[-1])-tf.log(self.num_samples),axis=0)

        return nELBO, prediction_m, prediction_std, test_loglik

    def train(self,mb_size,n_epochs=1,num_samples=1,lr=0.001):
        feed_dict={self.num_samples:num_samples,self.lr:lr}

        n_batches = int(np.ceil(1.0*self.N/mb_size))
        for ep in range(n_epochs):
            perm = np.random.permutation(self.N)
            for batch in range(n_batches):
                X_batch = self.X_nor[perm[np.arange(batch*mb_size,np.minimum((batch+1)*mb_size,self.N))],:]
                y_batch = self.y_nor[perm[np.arange(batch*mb_size,np.minimum((batch+1)*mb_size,self.N))],:]
                feed_dict.update({self.X_plh:X_batch, self.y_plh:y_batch})
                self.sess.run(self.train_op,feed_dict=feed_dict)
            self.trained_epochs += 1

    def evaluate(self,X_test,y_test,mb_size=500,num_samples=100):
        # test log lik, RMSE
        X_test_nor = np.zeros(X_test.shape)
        X_test_nor[:,self.std_X!=0] = (X_test[:,self.std_X!=0]-self.m_X[self.std_X!=0])/self.std_X[self.std_X!=0]
        y_test_nor = (y_test-self.m_y)/self.std_y

        n_batches = int(np.ceil(1.0*X_test.shape[0]/mb_size))
        test_loglik_pp, sqErr_pp = [], []
        for batch in range(n_batches):
            X_test_nor_batch = X_test_nor[np.arange(batch*mb_size,np.minimum((batch+1)*mb_size,X_test.shape[0])),:]
            y_test_nor_batch = y_test_nor[np.arange(batch*mb_size,np.minimum((batch+1)*mb_size,X_test.shape[0])),:]
            test_loglik_batch,prediction_m_batch = self.sess.run((self.test_loglik,self.prediction_m),
                                                            feed_dict={self.X_plh:X_test_nor_batch,
                                                                       self.y_plh:y_test_nor_batch,
                                                                       self.num_samples:num_samples})
            test_loglik_pp.append(test_loglik_batch)
            sqErr_pp.append((prediction_m_batch-y_test_nor_batch)**2)
        test_loglik_pp = np.concatenate(test_loglik_pp,axis=0)
        sqErr_pp = np.concatenate(sqErr_pp,axis=0)
        test_loglik = np.mean(test_loglik_pp)
        RMSE = np.sqrt(np.mean(sqErr_pp))
        return test_loglik-np.log(self.std_y), self.std_y*RMSE, test_loglik_pp-np.log(self.std_y), (self.std_y**2)*sqErr_pp

    def predict(self,X_test,mb_size=500,num_samples=100):
        X_test_nor = np.zeros(X_test.shape)
        X_test_nor[:,self.std_X!=0] = (X_test[:,self.std_X!=0]-self.m_X[self.std_X!=0])/self.std_X[self.std_X!=0]

        n_batches = int(np.ceil(1.0*X_test.shape[0]/mb_size))
        prediction_m, prediction_std = [],[]
        for batch in range(n_batches):
            X_test_nor_batch = X_test_nor[np.arange(batch*mb_size,np.minimum((batch+1)*mb_size,X_test.shape[0])),:]
            prediction_m_batch,prediction_std_batch = self.sess.run((self.prediction_m,self.prediction_std),
                                                            feed_dict={self.X_plh:X_test_nor_batch,
                                                                       self.num_samples:num_samples})
            prediction_m.append(prediction_m_batch)
            prediction_std.append(prediction_std_batch)
        prediction_m = np.concatenate(prediction_m,axis=0)
        prediction_std = np.concatenate(prediction_std,axis=0)
        return self.std_y*prediction_m+self.m_y, self.std_y*prediction_std

    def get_ELBO(self,mb_size=5000,num_samples=1):
        n_batches = int(np.ceil(1.0*self.N/mb_size))
        ELBO = 0.0
        ELBO_data = 0.0
        for batch in range(n_batches):
            X_batch = self.X_nor[np.arange(batch*mb_size,np.minimum((batch+1)*mb_size,self.N)),:]
            y_batch = self.y_nor[np.arange(batch*mb_size,np.minimum((batch+1)*mb_size,self.N)),:]
            nELBO,data_term,ELBO_KL = self.sess.run((self.nELBO,self.data_term,self.KL_term),feed_dict={self.X_plh:X_batch,
                                                              self.y_plh:y_batch,
                                                              self.num_samples:num_samples})
            ELBO += -1.0*nELBO
            ELBO_data += data_term
        return ELBO,ELBO_data,ELBO_KL
