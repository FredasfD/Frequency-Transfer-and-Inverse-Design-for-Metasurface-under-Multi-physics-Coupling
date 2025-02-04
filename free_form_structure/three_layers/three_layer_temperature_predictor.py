import numpy as np
import csv
import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,Dense,concatenate,Lambda,Add,Conv2D,MaxPooling2D,Flatten,Multiply
from keras.callbacks import ModelCheckpoint
data1 = np.loadtxt("dataset/G.dat",delimiter=',', dtype='float32')
data2 = np.loadtxt("dataset/S21.dat",delimiter=',', dtype='float32')
data3 = np.loadtxt("dataset/T.dat",delimiter=',', dtype='float32')
data1 = data1.reshape(np.size(data1,0),64,64)
num=46
dim=3

# const=0.01_0 dim=4 num=46

const=0.02
# maxx1=np.array([9.5,1.3,2.5,1.2])
# minx1=np.array([8,0.3,0.5,0.2])
# x1 = data1[:,0:4]
# x1=(x1-minx1)/(maxx1-minx1)*2-1
x2 = data2[:,0:num]*2-1
x3 = np.zeros((x2.shape[0],dim))
y1 = data3[:,0:num]
# train_num=np.size(data1,0)
train_num=600
x1_train=data1[0:train_num,:,:]
x2_train=x2[0:train_num,:]
x3_train=x3[0:train_num,:]
y1_train=y1[0:train_num,:]

x1_test=data1[train_num:,:]
x2_test=x2[train_num:,:]
x3_test=x3[train_num:,:]
y1_test=y1[train_num:,:]

# x1_test=data1[0:train_num,:,:]
# x2_test=x2[0:train_num,:]
# x3_test=x3[0:train_num,:]
# y1_test=y1[0:train_num,:]

x1_train = np.expand_dims(x1_train, axis=3)
x1_test = np.expand_dims(x1_test, axis=3)

in1=Input(shape=(1,))
in2=Input(shape=(dim,))
diy=120
a1 = concatenate([in1,in2])
a1=Dense(diy, kernel_initializer='he_normal', activation='relu',name='n1_l1')(a1)
a1=Dense(diy, kernel_initializer='he_normal', activation='relu',name='n1_l2')(a1)
a1=Dense(diy, kernel_initializer='he_normal', activation='relu',name='n1_l3')(a1)
a1=Dense(diy, kernel_initializer='he_normal', activation='relu',name='n1_l4')(a1)
a1=Dense(diy, kernel_initializer='he_normal', activation='relu',name='n1_l5')(a1)
a1=Dense(dim, kernel_initializer='uniform', activation='linear',name='n1_out')(a1)
# a1=Dense(dim, kernel_initializer='he_normal', activation='tanh',name='n1_out')(a1)
z1=Model(inputs=[in1,in2],outputs=a1)
z1.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
# z1.load_weights('D:/free_form/NN_1001_2024/model/temp_predictor/l1_weight_1.h5')

diy2=120
in3=Input(shape=(64,64,1,))
in4=Input(shape=(dim,))
in5=Input(shape=(1,))
b1=Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu')(in3)
b1=MaxPooling2D(pool_size=(2,2), strides=(2,2))(b1)
b1=Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu')(b1)
b1=MaxPooling2D(pool_size=(2,2), strides=(2,2))(b1)
b1=Conv2D(128,kernel_size=(5,5),strides=(1,1),activation='relu')(b1)
b1=MaxPooling2D(pool_size=(2,2), strides=(2,2))(b1)
b1=Flatten()(b1)
b1=Dense(1000, kernel_initializer='he_normal', activation='relu',name='n2_l1')(b1)
b1=Dense(500, kernel_initializer='he_normal', activation='relu',name='n2_l2')(b1)
b1=Dense(diy2, kernel_initializer='he_normal', activation='relu',name='n2_l3')(b1)
b1=Dense(diy2, kernel_initializer='he_normal', activation='relu',name='n2_l4')(b1)

b2=concatenate([in4,in5])
b2=Dense(diy2, kernel_initializer='he_normal', activation='relu',name='n3_l1')(b2)
b2=Dense(diy2, kernel_initializer='he_normal', activation='relu',name='n3_l2')(b2)
b2=Dense(diy2, kernel_initializer='he_normal', activation='relu',name='n3_l3')(b2)

b3=concatenate([b1,b2])

b3=Dense(diy2, kernel_initializer='he_normal', activation='relu',name='n3_l4')(b3)
b3=Dense(diy2, kernel_initializer='he_normal', activation='relu',name='n3_l5')(b3)
# b3=Dense(diy2, kernel_initializer='he_normal', activation='elu',name='n3_l6')(b3)
b3=Dense(1, kernel_initializer='uniform', activation='linear',name='n3_out')(b3)

z2=Model(inputs=[in3,in4,in5],outputs=b3)
# z2=Model(inputs=[in3,in4,in5],outputs=b5)
# z2=Model(inputs=[in3,in4],outputs=b3)
z2.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
# z2.load_weights('D:/free_form/NN_1001_2024/model/temp_predictor/l2_weight_1.h5')

inx1=Input(shape=(num,))
c1=Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': num})(inx1)
inx2=Input(shape=(dim,))
inx3=Input(shape=(64,64,1,))
t1=z1([c1[0],inx2])
t1=Lambda(lambda x: const*x)(t1)
d=z2([inx3,t1,c1[0]])
# d=z2([inx3,t1])
for i in range(1,num):
    t2=z1([c1[i],t1])
    t2=Lambda(lambda x: const*x)(t2)    
    t1=Add()([t1,t2])    
    d1=z2([inx3,t1,c1[i]])  
    # d1=z2([inx3,t1])
    d=concatenate([d,d1])    
    
m_combine=Model(inputs=[inx3,inx1,inx2], outputs=d)
m_combine.compile(loss='mse',optimizer='adam',metrics=['mse'])
filepath_1='model/combine_weight.h5'
filepath_2='model/combine_weight_fstep.h5'
filepath_3='model/l1_weight.h5'
filepath_4='model/l2_weight.h5'

callbacks = [
    ModelCheckpoint(filepath_1, monitor='loss', save_weights_only=True,
                    verbose=1,save_best_only=True,save_freq="epoch"),
]
train=False
if train:    
    history=m_combine.fit(x=[x1_train,x2_train,x3_train],
                          y=y1_train, epochs=5000, batch_size=100,shuffle=True, verbose=1,
                          validation_data=([x1_test,x2_test,x3_test], y1_test),callbacks=callbacks)
    loss_train=np.array(history.history['loss'])
    loss_test=np.array(history.history['val_loss'])
    m_combine.save_weights(filepath_2)
    m_combine.load_weights(filepath_1)
    z1.save_weights(filepath_3)
    z2.save_weights(filepath_4)
    x1 = data1
    x1 = np.expand_dims(data1, axis=3)
    x2 = data2[:,:]*2-1
    y1 = data3[:,:]    
    f1=z1.predict([x2[:,0],x3])*const
    # yy=z2.predict([x1,f1])    
    yy=z2.predict([x1,f1,x2[:,0]])    
    for i in range(1,61):
        f1=z1.predict([x2[:,i],f1])*const+f1        
        yy1=z2.predict([x1,f1,x2[:,i]])     
        # yy1=z2.predict([x1,f1])
        yy=np.c_[yy,yy1]
    e1=np.linalg.norm(yy[:,0:num]-y1[:,0:num],ord=2)/np.linalg.norm(y1[:,0:num],ord=2)
    e2=np.linalg.norm(yy[:,num:]-y1[:,num:],ord=2)/np.linalg.norm(y1[:,num:],ord=2)
else:
    x1 = data1
    x1 = np.expand_dims(data1, axis=3)
    x2 = data2[:,:]*2-1
    y1 = data3[:,:]
    z1.load_weights(filepath_3)
    z2.load_weights(filepath_4)
    f1=z1.predict([x2[:,0],x3])*const    
    yy=z2.predict([x1,f1,x2[:,0]])    
    for i in range(1,61):
        f1=z1.predict([x2[:,i],f1])*const+f1        
        yy1=z2.predict([x1,f1,x2[:,i]])        
        yy=np.c_[yy,yy1]
    # num=46
    e1=np.linalg.norm(yy[:,0:num]-y1[:,0:num],ord=2)/np.linalg.norm(y1[:,0:num],ord=2)
    e2=np.linalg.norm(yy[:,num:]-y1[:,num:],ord=2)/np.linalg.norm(y1[:,num:],ord=2)    
    # e3=np.zeros((1245,1))
    e3=np.zeros((446,1))
    resuty=10**yy
    simy=10**y1
    for i in range(0,446):
    # for i in range(0,1245):
        e3[i]=np.linalg.norm(10**yy[i,num:]-10**y1[i,num:],ord=2)/np.linalg.norm(10**y1[i,num:],ord=2)
        
    # with open("../results/temperature_predictor_result.csv","w",newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(yy)