import numpy as np
import matplotlib.pyplot as plt

import SimpleNN as snn

#method and point generation adapted from http://cs231n.github.io/neural-networks-case-study/

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

#generates the pointset

N = 100# number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
group = np.zeros(N*K, dtype='uint8') # group for in point
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  group[ix] = j

#set up the classifier
sfc = snn.SoftMaxClassifier(2,100,3)


for step in range(0,10000):
    if step%1000 == 0:
        print(step)
    sfc.train_step(X,group)
    

#plot the result

xx, yy = np.meshgrid(np.linspace(-1.5,1.5,50),np.linspace(-1.5,1.5,50))

gset = np.column_stack((xx.ravel(), yy.ravel()))
cvec = sfc.classify(gset)

Z=cvec.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.get_cmap('copper'), alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=group, s=40, cmap=plt.get_cmap('rainbow'))
plt.axis('equal')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5,1.5)   
plt.show()