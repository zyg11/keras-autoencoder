import numpy as np
from keras.models import Model
from keras.layers import Dense,Input,Lambda
from keras import backend as K
from  scipy.stats import norm
from keras import objectives
import matplotlib.pyplot as plt
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
batch_size=100
orginal_dim=784
latent_dim=2
intermediate_dim=256
nb_epoch=10
epsilon_std=1.0

#初始化编码器网络
#encoder
x=Input(batch_shape=(batch_size,orginal_dim))
h=Dense(intermediate_dim,activation='relu')(x)
z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)
print(z_mean)
print(z_log_var)

def sampling(args):
    z_mean,z_log_var=args
    epsilon=K.random_normal(shape=(batch_size,latent_dim),mean=0.)
    return z_mean+K.exp(z_log_var/2)*epsilon
#注意 the "output_shape" isn't necessary with the Tensorflow backend
z=Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_var])
#lantent hidden layer
print(z)
#decoder
decoder_h=Dense(intermediate_dim,activation='relu')
decoder_mean=Dense(orginal_dim,activation='sigmoid')
h_decoded=decoder_h(z)
x_decoded_mean=decoder_mean(h_decoded)
print(x_decoded_mean)

#loss
def vae_loss(x,x_decoder_mean):
    xent_loss=orginal_dim*objectives.binary_crossentropy(x,x_decoded_mean)
    kl_loss=-0.5*K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var),axis=-1)
    return xent_loss+kl_loss
vae=Model(x,x_decoded_mean)
vae.compile(optimizer='rmsprop',
            loss=vae_loss)#vae_loss
vae.fit(x_train,
        x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test,x_test))
#画出二维平面上的邻域。每个颜色聚类用一个数字表示，而闭合聚类本质上是与结构相似的数字。
#建立模型投影到二维空间
encoder=Model(x,z_mean)

x_test_encoded=encoder.predict(x_test,batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:,0],x_test_encoded[:,1],c=y_test)
plt.colorbar()
plt.show()