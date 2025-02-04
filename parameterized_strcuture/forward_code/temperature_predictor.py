import numpy as np
import csv
import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,Dense,concatenate,Lambda,Add
from keras.callbacks import ModelCheckpoint
data1 = np.loadtxt("../data/t-dataset/structure.csv",delimiter=',', dtype='float32')
data2 = np.loadtxt("../data/t-dataset/S_f.csv",delimiter=',', dtype='float32')
data3 = np.loadtxt("../data/t-dataset/temper.csv",delimiter=',', dtype='float32')
num=41
dim=2
const=0.05
maxx1=np.array([9.5,1.3,2.5,1.2])
minx1=np.array([8,0.3,0.5,0.2])
x1 = data1[:,0:4]
x1=(x1-minx1)/(maxx1-minx1)*2-1
x2 = data2[:,0:num]*2-1
x3 = np.zeros((x1.shape[0],dim))
y1 = data3[:,0:num]
train_num=650
x1_train=x1[0:train_num,:]
x2_train=x2[0:train_num,:]
x3_train=x3[0:train_num,:]
y1_train=y1[0:train_num,:]
x1_test=x1[train_num:,:]
x2_test=x2[train_num:,:]
x3_test=x3[train_num:,:]
y1_test=y1[train_num:,:]
in1=Input(shape=(1,))
in2=Input(shape=(dim,))
a1 = concatenate([in1,in2])
a1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n1_l1')(a1)
a1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n1_l2')(a1)
a1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n1_l3')(a1)
a1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n1_l4')(a1)
a1=Dense(dim, kernel_initializer='uniform', activation='linear',name='n1_out')(a1)
z1=Model(inputs=[in1,in2],outputs=a1)
z1.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
in3=Input(shape=(4,))
in4=Input(shape=(dim,))
in5=Input(shape=(1,))
b1=concatenate([in3,in4,in5])
b1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n2_l1')(b1)
b1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n2_l2')(b1)
b1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n2_l3')(b1)
b1=Dense(20, kernel_initializer='he_normal', activation='elu',name='n2_l4')(b1)
b1=Dense(1, kernel_initializer='uniform', activation='linear',name='n2_out')(b1)
z2=Model(inputs=[in3,in4,in5],outputs=b1)
z2.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
inx1=Input(shape=(num,))
c1=Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': num})(inx1)
inx2=Input(shape=(dim,))
inx3=Input(shape=(4,))
t1=z1([c1[0],inx2])
t1=Lambda(lambda x: const*x)(t1)
d=z2([inx3,t1,c1[0]])
for i in range(1,num):
    t2=z1([c1[i],t1])
    t2=Lambda(lambda x: const*x)(t2)    
    t1=Add()([t1,t2])    
    d1=z2([inx3,t1,c1[i]])    
    d=concatenate([d,d1])    
m_combine=Model(inputs=[inx3,inx1,inx2], outputs=d)
m_combine.compile(loss='mse',optimizer='adam',metrics=['mse'])
filepath_1='../data/t-model/model_combine_weight_3_o.h5'
filepath_2='../data/t-model/model_combine_finalstep_weight_3_o.h5'
filepath_3='../data/t-model/model_l1_weight_3_o.h5'
filepath_4='../data/t-model/model_l2_weight_3_o.h5'
callbacks = [
    ModelCheckpoint(filepath_1, monitor='val_loss', save_weights_only=True,verbose=1,save_best_only=True,period=1),
]
train=False
if train:    
    history=m_combine.fit(x=[x1_train,x2_train,x3_train],
                          y=y1_train, epochs=10000, batch_size=100,shuffle=True, verbose=1,
                          validation_data=([x1_test,x2_test,x3_test], y1_test),callbacks=callbacks)
    loss_train=np.array(history.history['loss'])
    loss_test=np.array(history.history['val_loss'])
    m_combine.save_weights(filepath_2)
    m_combine.load_weights(filepath_1)
    z1.save_weights(filepath_3)
    z2.save_weights(filepath_4)
    x1 = data1[:,0:4]
    x1=(x1-minx1)/(maxx1-minx1)*2-1
    x2 = data2[:,:]*2-1
    y1 = data3[:,:]    
    f1=z1.predict([x2[:,0],x3])*const    
    yy=z2.predict([x1,f1,x2[:,0]])    
    for i in range(1,61):
        f1=z1.predict([x2[:,i],f1])*const+f1        
        yy1=z2.predict([x1,f1,x2[:,i]])        
        yy=np.c_[yy,yy1]
    e1=np.linalg.norm(yy[:,0:num]-y1[:,0:num],ord=2)/np.linalg.norm(y1[:,0:num],ord=2)
    e2=np.linalg.norm(yy[:,num:]-y1[:,num:],ord=2)/np.linalg.norm(y1[:,num:],ord=2)
else:
    x1 = data1[:,0:4]
    x1=(x1-minx1)/(maxx1-minx1)*2-1
    x2 = data2[:,:]*2-1
    y1 = data3[:,:]
    z1.load_weights(filepath_3)
    z2.load_weights(filepath_4)
    f1=z1.predict([x2[:,0],x3])*const    
    yy=z2.predict([x1,f1,x2[:,0]])    
    fpo=np.zeros((61,2))
    fpo[0]=np.mean(f1,axis=0)
    fa=np.zeros((61*710,2))
    fa[0:710]=f1
    for i in range(1,61):
        f1=z1.predict([x2[:,i],f1])*const+f1     
        fpo[i]=np.mean(f1,axis=0)
        fa[i*710:i*710+710]=f1
        yy1=z2.predict([x1,f1,x2[:,i]])        
        yy=np.c_[yy,yy1]
    e1=np.linalg.norm(yy[:,0:num]-y1[:,0:num],ord=2)/np.linalg.norm(y1[:,0:num],ord=2)
    e2=np.linalg.norm(yy[:,num:]-y1[:,num:],ord=2)/np.linalg.norm(y1[:,num:],ord=2)    
    print('relative error in seen frequency:')
    print(e1)
    print('relative error for frequency transfer:')
    print(e2)