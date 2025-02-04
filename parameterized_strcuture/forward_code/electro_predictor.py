import os
os.environ["DDE_BACKEND"] = "tensorflow"
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
d1 = np.loadtxt("../data/em-dataset/x_e_train_x_S.csv",delimiter=',', dtype='float32')
d2 = np.loadtxt("../data/em-dataset/y_e_train_x_S.csv",delimiter=',', dtype='float32')
d3 = np.loadtxt("../data/em-dataset/x_e_test_x_S.csv",delimiter=',', dtype='float32')
d4 = np.loadtxt("../data/em-dataset/y_e_test_x_S.csv",delimiter=',', dtype='float32')
x1 = d1[:,0:4]
x2 = d1[:,4:5]
x3 = d3[:,0:4]
x4 = d3[:,4:5]
y=d2[:,1:2]
y2=d4[:,1:2]
maxx1=np.max(x1,axis=0)
minx1=np.min(x1,axis=0)
maxx2=np.max(x2,axis=0)
minx2=np.min(x2,axis=0)
maxy=np.max(y,axis=0)
miny=np.min(y,axis=0)
x1=(x1-minx1)/(maxx1-minx1)*2-1
x2=(x2-minx2)/(maxx2-minx2)*2-1
x3=(x3-minx1)/(maxx1-minx1)*2-1
x4=(x4-minx2)/(maxx2-minx2)*2-1
y=y*2-1
y2=y2*2-1
X_train = (x1, x2)
y_train = y
X_test = (x3, x4)
y_test = y2
### this part is from the demo of DeepXDE
data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
m = 4
dim_x = 1
net = dde.nn.DeepONet(
    [m, 300, 300, 300, 300],
    [dim_x, 300, 300, 300, 300],
    "relu",
    "Glorot normal",
)
# Define a Model
model = dde.Model(data, net)
# Compile and Train
model.compile("adam", lr=0.001, loss='MSE')
##############################################
train=False
if train:
    losshistory, train_state = model.train(iterations=30000,display_every=500,batch_size=10000)
    # Plot the loss trajectory
    dde.utils.plot_loss_history(losshistory)
    plt.show()
    model.save("../data/em-model\\m_300_4", protocol="backend")
else:
    model.restore("../data/em-model\\m_300_4-40000.ckpt")
xx=model.predict(X_test)
l1=np.linalg.norm(xx-y_test,ord=2)/np.linalg.norm(y_test,ord=2)
er=xx-y_test
i=np.argmax(er)
print('relative error:')
print(l1)