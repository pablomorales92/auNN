import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
sys.path.append("code/")

os.environ["CUDA_VISIBLE_DEVICES"]=""

tf.set_random_seed(1)
np.random.seed(1)

data = np.load("train_data.npz")
x_tr = data['x_train']
y_tr = data['y_train']
x_ts = np.linspace(-5,5,500)[:,None]

from auNN import auNN
arch = [25,1]
M = [10,10]
model_RBF = auNN(x_tr,y_tr,arch,M,ker='RBF',init_prior_l=0.1)
model_TRI = auNN(x_tr,y_tr,arch,M,ker='TRI',init_prior_l=0.1)

for i in range(5000):
    if (i+1)%100==0:
        print("{}/5000".format(i+1))
    model_RBF.train(mb_size=x_tr.shape[0])
    model_TRI.train(mb_size=x_tr.shape[0])


plt.figure(figsize=(5,2))
plt.subplot(121)
plt.title("auNN-RBF")
pred_m, pred_std = model_RBF.predict(x_ts)
plt.plot(x_ts.flatten(),pred_m.flatten(),color="skyblue")
plt.fill_between(x_ts.flatten(),
                 (pred_m-pred_std).flatten(),
                 (pred_m+pred_std).flatten(),
                 alpha=0.4,color="skyblue")
plt.scatter(x_tr,y_tr,color="black")
plt.subplot(122)
plt.title("auNN-TRI")
pred_m, pred_std = model_TRI.predict(x_ts)
plt.plot(x_ts.flatten(),pred_m.flatten(),color="blue")
plt.fill_between(x_ts.flatten(),
                 (pred_m-pred_std).flatten(),
                 (pred_m+pred_std).flatten(),
                 alpha=0.4,color="blue")
plt.scatter(x_tr,y_tr,color="black")

plt.tight_layout()
plt.savefig("fig.png")
