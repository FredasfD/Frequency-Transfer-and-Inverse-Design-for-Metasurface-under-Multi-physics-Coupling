import os
os.environ["DDE_BACKEND"] = "tensorflow"
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
d1 = np.loadtxt("../data/e-t-dataset/x_et_train_x_S.csv",delimiter=',', dtype='float32')
d2 = np.loadtxt("../data/e-t-dataset/y_et_train_x_S.csv",delimiter=',', dtype='float32')
d3 = np.loadtxt("../data/e-t-dataset/x_et_test_x_S.csv",delimiter=',', dtype='float32')
d4 = np.loadtxt("../data/e-t-dataset/y_et_test_x_S.csv",delimiter=',', dtype='float32')
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
x1=(x1-minx1)/(maxx1-minx1)*2-1
x2_1=(x2-0.1-minx2)/(maxx2-minx2)*2-1
x2_2=(x2-minx2)/(maxx2-minx2)*2-1
x2_3=(x2+0.1-minx2)/(maxx2-minx2)*2-1
x3=(x3-minx1)/(maxx1-minx1)*2-1
x4_1=(x4-0.1-minx2)/(maxx2-minx2)*2-1
x4_2=(x4-minx2)/(maxx2-minx2)*2-1
x4_3=(x4+0.1-minx2)/(maxx2-minx2)*2-1
y=y*2-1
y2=y2*2-1
X_train_11 = (x1, x2_1)
X_train_12 = (x1, x2_2)
X_train_13 = (x1, x2_3)
y_2_train = y
X_test_11 = (x3, x4_1)
X_test_12 = (x3, x4_2)
X_test_13 = (x3, x4_3)
y_2_test = y2
### this part is from the demo of DeepXDE
data1 = dde.data.Triple(X_train=X_train_12, y_train=y_2_train, X_test=X_test_12, y_test=y_2_test)
# Choose a network
m1 = 4
dim_x1 = 1
net1 = dde.nn.DeepONet(
    [m1, 300, 300, 300, 300],
    [dim_x1, 300, 300, 300, 300],
    "relu",
    "Glorot normal",
)
# Define a Model
model1 = dde.Model(data1, net1)
# Compile and Train
model1.compile("adam", lr=0.001, loss='MSE')
##############################################
dir1="..\\data\\em-model\\"
model1.restore(dir1+".\\"+"m_300_4-40000.ckpt")
y_1_train1=model1.predict(X_train_11)
y_1_train2=model1.predict(X_train_12)
y_1_train3=model1.predict(X_train_13)
y_1_test1=model1.predict(X_test_11)
y_1_test2=model1.predict(X_test_12)
y_1_test3=model1.predict(X_test_13)
y_train=y_2_train-y_1_train2
y_test=y_2_test-y_1_test2
maxy=np.max(y_train,axis=0)
miny=np.min(y_train,axis=0)
x_2_train=np.c_[x2_2,y_1_train1]
x_2_train=np.c_[x_2_train,y_1_train2]
x_2_train=np.c_[x_2_train,y_1_train3]
x_2_test=np.c_[x4_2,y_1_test1]
x_2_test=np.c_[x_2_test,y_1_test2]
x_2_test=np.c_[x_2_test,y_1_test3]
X_train = (x1, x_2_train)
X_test = (x3, x_2_test)
### this part is from the demo of DeepXDE
data2 = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
m2 = 4
dim_x2 = 4
net2 = dde.nn.DeepONet(
    [m2, 300, 300, 300],
    [dim_x2, 300, 300],
    "relu",
    "Glorot normal",
    regularization = ["l2", 0.01]
)
model2 = dde.Model(data2, net2)
model2.compile("adam", lr=0.001, loss='MSE')
##############################################

train=False
if train:
    losshistory, train_state = model2.train(iterations=1000,display_every=10)
    # Plot the loss trajectory
    dde.utils.plot_loss_history(losshistory)
    plt.show()
    model2.save("..\data\e-t-model\m_re_1", protocol="backend")
else:
    model2.restore("..\data\e-t-model\\m_re_1-1000.ckpt")
xx=model2.predict(X_test)
results=xx+y_1_test2
l2=np.linalg.norm(xx+y_1_test2-y_2_test,ord=2)/np.linalg.norm(y_2_test,ord=2)
er1=xx-y_test

print('relative error:')
print(l2)