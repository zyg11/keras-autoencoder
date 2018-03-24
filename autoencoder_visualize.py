from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import numpy as np
import matplotlib.pyplot as plt

(x_train,_),(x_test,y_test)=mnist.load_data()
x_train=x_train.astype('float32')/255. -0.5
x_test=x_test.astype('float32')/255. -0.5
x_train=x_train.reshape(len(x_train),-1)
x_test=x_test.reshape(len(x_test),-1)
print(x_train.shape)
print(x_test.shape)
# 压缩特征维度至2维
encoding_dim=2
input_img=Input(shape=(784,))
#编码器
encoded=Dense(128,activation='relu')(input_img)
encoded=Dense(64,activation='relu')(encoded)
encoded=Dense(10,activation='relu')(encoded)
encoder_output=Dense(encoding_dim)(encoded)
#解码
decoded=Dense(10,activation='relu')(encoder_output)
decoded=Dense(64,activation='relu')(decoded)
decoded=Dense(128,activation='relu')(decoded)
decoded=Dense(784,activation='tanh')(decoded)
# decoded=Dense(2,activation='sigmoid')(decoded)
#构建自编码器
autoencoder=Model(inputs=input_img,outputs=decoded)
#构建编码模型
encoder=Model(inputs=input_img,outputs=encoder_output)
autoencoder.compile(optimizer='adam',
                    loss='mse',
                    metrics=['accuracy'])
autoencoder.fit(
    x_train,
    x_train,
    batch_size=128,
    epochs=20,
    shuffle=True,
    # validation_data=(x_test,y_test)
)

encodered_imgs=encoder.predict(x_test)
plt.scatter(encodered_imgs[:,0],encodered_imgs[:,1],c=y_test, s=3)
plt.colorbar()
plt.show()