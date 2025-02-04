# import tensorflow as tfs
import os
os.environ["DDE_BACKEND"] = "tensorflow"
import deepxde as dde
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input,Dense,concatenate,Lambda,Add,Multiply
from keras.callbacks import ModelCheckpoint
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=None)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

d1 = np.loadtxt("dataset/x_et_train_x_S.csv",delimiter=',', dtype='float32')
d2 = np.loadtxt("dataset/y_et_train_x_S.csv",delimiter=',', dtype='float32')
d3 = np.loadtxt("dataset/x_et_test_x_S.csv",delimiter=',', dtype='float32')
d4 = np.loadtxt("dataset/y_et_test_x_S.csv",delimiter=',', dtype='float32')
x1 = d1[:,0:4]
x2 = d1[:,4:5]
x3 = d3[:,0:4]
x4 = d3[:,4:5]
y=d2[:,1:2]
y2=d4[:,1:2]
# maxx1=np.max(x1,axis=0)
# minx1=np.min(x1,axis=0)
maxx1=np.array([9.5,1.3,2.5,1.2])
minx1=np.array([8,0.3,0.5,0.2])
maxx2=14
minx2=2
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
data1 = dde.data.Triple(X_train=X_train_12, y_train=y_2_train, X_test=X_test_12, y_test=y_2_test)
# electro-model
### this part is from the demo of DeepXDE
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
model1.restore("em-predictor\model\m_300_4-40000.ckpt")
net1.trainable=False
Inputx=Input(shape=(4,))
Inputf2=Input(shape=(1,))
Inputf1=Lambda(lambda x: x-0.1/(maxx2-minx2)*2)(Inputf2)
Inputf3=Lambda(lambda x: x+0.1/(maxx2-minx2)*2)(Inputf2)
lab1=net1.call((Inputx,Inputf1),training=False)
lab2=net1.call((Inputx,Inputf2),training=False)
lab3=net1.call((Inputx,Inputf3),training=False)
# electro-thermal model data
y_1_train1=model1.predict(X_train_11)
y_1_train2=model1.predict(X_train_12)
y_1_train3=model1.predict(X_train_13)
y_1_test1=model1.predict(X_test_11)
y_1_test2=model1.predict(X_test_12)
y_1_test3=model1.predict(X_test_13)
y_train=y_2_train-y_1_train2
y_test=y_2_test-y_1_test2
x_2_train=np.c_[x2_2,y_1_train1]
x_2_train=np.c_[x_2_train,y_1_train2]
x_2_train=np.c_[x_2_train,y_1_train3]
x_2_test=np.c_[x4_2,y_1_test1]
x_2_test=np.c_[x_2_test,y_1_test2]
x_2_test=np.c_[x_2_test,y_1_test3]
X_train = (x1, x_2_train)
X_test = (x3, x_2_test)
data2 = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
# electro-thermal model
### this part is from the demo of DeepXDE
m2 = 4
dim_x2 = 4
net2 = dde.nn.DeepONet(
    [m2, 300, 300, 300],
    [dim_x2, 300, 300],
    "relu",
    "Glorot normal",
    # regularization = ["l2", 0.01]
)
net2.trainable=False
model2 = dde.Model(data2, net2)
##############################################
model2.restore("multi-predictor\model\m_re_1-1000.ckpt")
combine = concatenate([Inputf2,lab1,lab2,lab3])
Out1=net2.call((Inputx,combine),training=False)
Out1=Add()([Out1,lab2])
model_v = Model(inputs=[Inputx,Inputf2], outputs=Out1)
# model_v.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
model_v.trainable=False


# inverse model
inS=Input(shape=(61,))
inT=Input(shape=(2,))
# ins=concatenate([inS,inT]) 
a1=Dense(120, kernel_initializer='he_normal', activation='elu')(inS)
a1=Dense(120, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(60, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(60, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(20, kernel_initializer='he_normal', activation='elu')(a1)
a1=Dense(20, kernel_initializer='he_normal', activation='elu')(a1)

a2=Dense(20, kernel_initializer='he_normal', activation='elu')(inT)
a2=Dense(20, kernel_initializer='he_normal', activation='elu')(a2)
a2=Dense(20, kernel_initializer='he_normal', activation='elu')(a2)
a3=concatenate([a1,a2]) 

b1 = Dense(1,kernel_initializer='uniform', activation='tanh')(a3)
b2 = Dense(1,kernel_initializer='uniform', activation='tanh')(a3)
b3 = Dense(1,kernel_initializer='uniform', activation='tanh')(a3)
b4 = Dense(1,kernel_initializer='uniform', activation='tanh')(a3)
b = concatenate([b1,b2,b3,b4])
# z1=Model(inputs=inS,outputs=b)
z1=Model(inputs=[inS,inT],outputs=b)
z1.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
# physic val
c1=Lambda(lambda x: 0.06923612-0.34910977*x)(b1)
c2=Lambda(lambda x: 0.30008748*x)(b2)
c3=Lambda(lambda x: 0.18713269*x)(b3)
c4=Lambda(lambda x: 0.19797382*x)(b4)
c5=Lambda(lambda x: -0.04959737*x**2)(b1)
c6=Multiply()([b1, b2])
c6=Lambda(lambda x: -0.01124296*x)(c6)
c7=Multiply()([b1, b3])
c7=Lambda(lambda x: 0.03403281*x)(c7)
c8=Multiply()([b1, b4])
c8=Lambda(lambda x: -0.06397589*x)(c8)
c9=Lambda(lambda x: -0.02223586*x**2)(b2)
c10=Multiply()([b2, b3])
c10=Lambda(lambda x: 0.02517629*x)(c10)
c11=Multiply()([b2, b4])
c11=Lambda(lambda x: 0.01048303*x)(c11)
c12=Lambda(lambda x: 0.01389211*x**2)(b3)
c13=Multiply()([b3, b4])
c13=Lambda(lambda x: 0.0635559*x)(c13)
c14=Lambda(lambda x: -0.02373008*x**2)(b4)
c=Add()([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14])


d1=Lambda(lambda x: -0.29518393-0.2797401*x)(b1)
d2=Lambda(lambda x: 0.433523*x)(b2)
d3=Lambda(lambda x: 0.12549168*x)(b3)
d4=Lambda(lambda x: -0.03495054*x)(b4)
d5=Lambda(lambda x: 0.00355317*x**2)(b1)
d6=Multiply()([b1, b2])
d6=Lambda(lambda x: -0.06096978*x)(d6)
d7=Multiply()([b1, b3])
d7=Lambda(lambda x: -0.04154302*x)(d7)
d8=Multiply()([b1, b4])
d8=Lambda(lambda x: 0.02724634*x)(d8)
d9=Lambda(lambda x: -0.00246469*x**2)(b2)
d10=Multiply()([b2, b3])
d10=Lambda(lambda x: 0.05212497*x)(d10)
d11=Multiply()([b2, b4])
d11=Lambda(lambda x: -0.00630015*x)(d11)
d12=Lambda(lambda x: 0.09048595*x**2)(b3)
d13=Multiply()([b3, b4])
d13=Lambda(lambda x: -0.00455212*x)(d13)
d14=Lambda(lambda x: 0.01280437*x**2)(b4)
d=Add()([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14])
cd=concatenate([c,d])

# forward val
fstep=0.4
in1=Input(shape=(1,))
out_v=model_v([z1.output,in1])
ini=Lambda(lambda x: x+1*fstep/(maxx2-minx2)*2)(in1)
for i in range(1,31):
    outvi=model_v([z1.output,ini])
    out_v=concatenate([out_v,outvi]) 
    ini=Lambda(lambda x: x+1*fstep/(maxx2-minx2)*2)(ini)
    


# temp predictor
dim=3
const=0.05
inS21=Input(shape=(1,))
inz=Input(shape=(dim,))
diy=100
num=31
le1 = concatenate([inS21,inz])
le1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n1_l1')(le1)
le1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n1_l2')(le1)
le1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n1_l3')(le1)
le1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n1_l4')(le1)
# a1=Dense(dim, kernel_initializer='uniform', activation='linear',name='n1_out')(a1)
le1=Dense(dim, kernel_initializer='he_normal', activation='tanh',name='n1_out')(le1)
z2=Model(inputs=[inS21,inz],outputs=le1)
z2.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
z2.load_weights('thermal-predictor\model1\model_l1_weight_3_o.h5')
z2.trainable=False

inz3=Input(shape=(4,))
inz4=Input(shape=(dim,))
inz5=Input(shape=(1,))
lb1=concatenate([inz3,inz4,inz5])
lb1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n2_l1')(lb1)
lb1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n2_l2')(lb1)
lb1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n2_l3')(lb1)
lb1=Dense(diy, kernel_initializer='he_normal', activation='elu',name='n2_l4')(lb1)
lb1=Dense(1, kernel_initializer='uniform', activation='linear',name='n2_out')(lb1)
z3=Model(inputs=[inz3,inz4,inz5],outputs=lb1)
z3.compile(loss='mse',optimizer='adam',metrics=['accuracy','mse','mae'])
z3.load_weights('thermal-predictor\model1\model_l2_weight_3_o.h5')
z3.trainable=False



lc1=Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 61})(inS)
iny2=Input(shape=(dim,))
iny3=b
t1=z2([lc1[0],iny2])
t1=Lambda(lambda x: const*x)(t1)
td=z3([iny3,t1,lc1[0]])
for i in range(1,num):
    t2=z2([lc1[i],t1])
    t2=Lambda(lambda x: const*x)(t2)    
    t1=Add()([t1,t2])    
    td1=z3([iny3,t1,lc1[i]])    
    td=concatenate([td,td1])

# m_combine=Model(inputs=[inz3,inx1,inx2], outputs=d)


# m_combine=Model(inputs=[inS,in1,iny2], outputs=[b,cd,out_v,td])
m_combine=Model(inputs=[inS,in1,iny2,inT], outputs=[b,cd,out_v,td])
m_combine.compile(loss=['mse','mse','mse','mse'],
                   loss_weights=[0,0,1,0],
                  # loss_weights=[1,0,0],
                  optimizer='adam',
                  metrics=['mse'])
train=False
# # case1
# filepath_1='model/model_combine_weight_1_0_0_0.h5'
# filepath_2='model/model_inverse_weight_1_0_0_0.h5'
# filepath_3='model/model_combine_finalstep_weight_1_0_0_0.h5'
# model_name='model/inverse_model_1_0_0_0.keras'

# # case2
# filepath_1='model/model_combine_weight_0.2_0.8_0_0.h5'
# filepath_2='model/model_inverse_weight_0.2_0.8_0_0.h5'
# filepath_3='model/model_combine_finalstep_weight_0.2_0.8_0_0.h5'
# model_name='model/inverse_model_0.2_0.8_0_0.keras'

# # case3
# filepath_1='model/model_combine_weight_0.3_0_0.7_0.h5'
# filepath_2='model/model_inverse_weight_0.3_0_0.7_0.h5'
# filepath_3='model/model_combine_finalstep_weight_0.3_0_0.7_0.h5'
# model_name='model/inverse_model_0.3_0_0.7_0.keras'

# # case4
# filepath_1='model/model_combine_weight_0.2_0_0_0.8.h5'
# filepath_2='model/model_inverse_weight_0.2_0_0_0.8.h5'
# filepath_3='model/model_combine_finalstep_weight_0.2_0_0_0.8.h5'
# model_name='model/inverse_model_0.2_0_0_0.8.keras'

# # case5
filepath_1='model/model_combine_weight_0.15_0.5_0.35_0.h5'
filepath_2='model/model_inverse_weight_0.15_0.5_0.35_0.h5'
filepath_3='model/model_combine_finalstep_weight_0.15_0.5_0.35_0.h5'
model_name='model/inverse_model_0.15_0.5_0.35_0.keras'

# # case6
# filepath_1='model/model_combine_weight_0.2_0.4_0_0.4.h5'
# filepath_2='model/model_inverse_weight_0.2_0.4_0_0.4.h5'
# filepath_3='model/model_combine_finalstep_weight_0.2_0.4_0_0.4.h5'
# model_name='model/inverse_model_0.2_0.4_0_0.4.keras'

# case7
# filepath_1='model/model_combine_weight_0.1_0.4_0.2_0.3.h5'
# filepath_2='model/model_inverse_weight_0.1_0.4_0.2_0.3.h5'
# filepath_3='model/model_combine_finalstep_weight_0.1_0.4_0.2_0.3.h5'
# model_name='model/inverse_model_0.1_0.4_0.2_0.3.keras'

callbacks = [
    ModelCheckpoint(filepath_1, monitor='val_loss', save_weights_only=True,verbose=1,save_best_only=True,period=1),
]
dataset1=np.loadtxt("dataset/inverse_design.csv",delimiter=',', dtype='float32')
dataset2=np.loadtxt("dataset/deeponet_val.csv",delimiter=',', dtype='float32')
dataset3=np.loadtxt("dataset/reg.csv",delimiter=',', dtype='float32')
dataset4=np.loadtxt("dataset/temp.csv",delimiter=',', dtype='float32')
dataset5=np.loadtxt("dataset/temp_loc.csv",delimiter=',', dtype='float32')
stru=(dataset1[:,0:4]-minx1)/(maxx1-minx1)*2-1
s21=dataset1[:,4:65]*2-1

# f_sample=(dataset2[:,0:61]-minx2)/(maxx2-minx2)*2-1
# S_sample=dataset2[:,61:122]*2-1
f_sample=(dataset2[:,0:31]-minx2)/(maxx2-minx2)*2-1
S_sample=dataset2[:,31:62]*2-1
xz_0 = np.zeros((stru.shape[0],dim))
# fy=dataset3[:,4].reshape(len(dataset3[:,4]),1)
fy=dataset3[:,4:6]
train_num=635
temp=dataset4[:,4:35]
t_loc=dataset5[:,4:5]/(31)*2-1
t_max=(dataset5[:,5:6]-1)/2*2-1
t_position=np.c_[t_loc,t_max]
# train
stru_1=stru[0:train_num,:]
S21_1=s21[0:train_num,:]
f_1=f_sample[0:train_num,:]
S_1=S_sample[0:train_num,:]
fy1=fy[0:train_num,:]
xz_0_train = np.zeros((stru_1.shape[0],dim))
temp_1=temp[0:train_num,:]
tp_1=t_position[0:train_num,:]
# val
stru_2=stru[train_num:710,:]
S21_2=s21[train_num:710,:]
f_2=f_sample[train_num:710,:]
S_2=S_sample[train_num:710,:]
fy2=fy[train_num:710,:]
xz_0_test = np.zeros((stru_2.shape[0],dim))
temp_2=temp[train_num:710,:]
tp_2=t_position[train_num:710,:]
if train:
    # history=m_combine.fit(x=[S21_1,f_1[:,0],xz_0_train],
    #                       y=[stru_1,fy1,S_1,temp_1], epochs=100, batch_size=100, shuffle=True, verbose=1,
    #                       validation_data=([S21_2,f_2[:,0],xz_0_test], [stru_2,fy2,S_2,temp_2]),
    #                       callbacks=callbacks)
    history=m_combine.fit(x=[S21_1,f_1[:,0],xz_0_train,tp_1],
                          y=[stru_1,fy1,S_1,temp_1], epochs=500, batch_size=100, shuffle=True, verbose=1,
                          validation_data=([S21_2,f_2[:,0],xz_0_test,tp_2], [stru_2,fy2,S_2,temp_2]),
                          callbacks=callbacks)
    loss_train=np.array(history.history['loss'])
    loss_test=np.array(history.history['val_loss'])
    m_combine.save_weights(filepath_3)
    m_combine.load_weights(filepath_1)
    z1.save_weights(filepath_2)    
    z1.save(model_name)    
    # # m_combine.load_weights(filepath_3)
    # testset=np.loadtxt("dataset/inverse_test_2.csv",delimiter=',', dtype='float32')    
    # test=testset[:,1].reshape(1,len(testset[:,1]))*2-1
    # testy=(z1.predict(test,temp_2)+1)/2*(maxx1-minx1)+minx1
    # # model.save("G:\electro-thermal\electro\model_total", protocol="backend")
    [yy,ff,ss,tt]= m_combine.predict([S21_2,f_2[:,0],xz_0_test,tp_2])
    y_true=stru_2
    s_true=S_2
    ff_true=fy2
    e1=np.linalg.norm(yy-y_true,ord=2)/np.linalg.norm(y_true,ord=2)    
    e2=np.linalg.norm(ss-s_true,ord=2)/np.linalg.norm(s_true,ord=2)
    # e3=np.linalg.norm(ff-ff_true,ord=2)/np.linalg.norm(ff_true,ord=2)
    ff=(ff+1)*6+2
    ff_true=(ff_true+1)*6+2
    e3=np.linalg.norm(ff[:,0]-ff_true[:,0],ord=2)/np.linalg.norm(ff_true[:,0],ord=2)
    e4=np.linalg.norm(ff[:,1]-ff_true[:,1],ord=2)/np.linalg.norm(ff_true[:,1],ord=2)
    e5=np.linalg.norm(tt-temp_2,ord=2)/np.linalg.norm(temp_2,ord=2)
else:  
    z1.load_weights(filepath_2)    
    testset=np.loadtxt("dataset/inverse_test_2.csv",delimiter=',', dtype='float32')    
    test=testset[:,1].reshape(1,len(testset[:,1]))*2-1
    # testy=(z1.predict(test)+1)/2*(maxx1-minx1)+minx1
    
    [yy,ff,ss,tt]= m_combine.predict([S21_2,f_2[:,0],xz_0_test,tp_2])
    
    f_val=f_2[:,0]
    val_S=model_v.predict([yy,f_val])
    for i in range(1,61):
        f_val=f_val+0.2/(maxx2-minx2)*2
        val_S21=model_v.predict([yy,f_val])
        val_S=np.c_[val_S,val_S21]        
    val_S=(val_S+1)/2
    
    
    f_val=f_2[:,0]
    val_S=model_v.predict([yy,f_val])
    for i in range(1,61):
        f_val=f_val+0.2/(maxx2-minx2)*2
        val_S21=model_v.predict([yy,f_val])
        val_S=np.c_[val_S,val_S21]        
    val_S=(val_S+1)/2
    
    # forward val
    fstep2=0.2
    f_start=f_2[:,0]
    s_p=model_v.predict([yy,f_start])
    for i in range(1,31):
        f_start=f_start+1*fstep2/(maxx2-minx2)*2
        s_predict=model_v.predict([yy,f_start])
        s_p=np.c_[s_p,s_predict]        
    
    f1=z2.predict([s_p[:,0],xz_0_test])*const
    t_p=z3.predict([yy,f1,s_p[:,0]])    
    for i in range(1,31):
        f1=z2.predict([s_p[:,i],f1])*const+f1        
        t_p1=z3.predict([yy,f1,s_p[:,i]])        
        t_p=np.c_[t_p,t_p1]
    
    
    y_true=stru_2
    s_true=S_2
    ff_true=fy2
    E1=np.linalg.norm(yy-y_true,ord=2)/np.linalg.norm(y_true,ord=2)    
    E3=np.linalg.norm(ss-s_true,ord=2)/np.linalg.norm(s_true,ord=2)
    # e3=np.linalg.norm(ff-ff_true,ord=2)/np.linalg.norm(ff_true,ord=2)
    ff=(ff+1)*6+2
    ff_true=(ff_true+1)*6+2
    E21=np.linalg.norm(ff[:,0]-ff_true[:,0],ord=2)/np.linalg.norm(ff_true[:,0],ord=2)
    E22=np.linalg.norm(ff[:,1]-ff_true[:,1],ord=2)/np.linalg.norm(ff_true[:,1],ord=2)
    # tt=10**(tt)
    # temp_2=10**(temp_2)
    # e5=np.linalg.norm(tt-temp_2,ord=2)/np.linalg.norm(temp_2,ord=2)
    E4=np.linalg.norm(t_p-temp_2,ord=2)/np.linalg.norm(temp_2,ord=2)
    
    yy_p=(yy+1)/2*(maxx1-minx1)+minx1
    yy_true=(y_true+1)/2*(maxx1-minx1)+minx1
    print('[E1,E21,E22,E3,E4]')
    print([E1,E21,E22,E3,E4])
    
