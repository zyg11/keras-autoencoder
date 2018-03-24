from keras.models import Model
from keras.models import load_model
import  numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
(x_train,_),(x_test,_)=mnist.load_data()
number=10000
x_train=x_train[0:number]
x_test=x_test[0:5000]
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train=np.reshape(x_train,(len(x_train),28,28,1))
x_test=np.reshape(x_test,(len(x_test),28,28,1))

autoencoder=load_model('E:/keras_data/autoencoder/CNN_autocoder.h5')
autoencoder.summary()
model_extractfeatures=Model(inputs=autoencoder.input,outputs=autoencoder.get_layer('conv2d_7').output)
encoderded_imgs=model_extractfeatures.predict(x_test)
print(encoderded_imgs.shape)
n=10
plt.figure(figsize=(20,8))
for i in range(n):
    ax=plt.subplot(1,n,i+1)
    plt.imshow(encoderded_imgs[i].reshape(28,1*28).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()