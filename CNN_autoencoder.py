from keras.models import Model
from keras.layers import Dense,Conv2D,UpSampling2D,MaxPooling2D,Input
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train,_),(x_test,_)=mnist.load_data()
number=10000
x_train = x_train[0:number]
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train=np.reshape(x_train,(len(x_train),28,28,1))
x_test=np.reshape(x_test,(len(x_test),28,28,1))
#convolutional autoencoder
input_img=Input(shape=(28,28,1))
x = Conv2D(16, (3, 3), activation='relu',padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2),padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x=Conv2D(8,(3,3),activation='relu', padding='same')(encoded)
x=UpSampling2D((2,2))(x)
x=Conv2D(8,(3,3),activation='relu', padding='same')(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(16,(3,3),activation='relu')(x)#有问题
x=UpSampling2D((2,2))(x)
decoded=Conv2D(1,(3,3),activation='sigmoid' ,padding='same')(x)


autoencoder=Model(input_img,decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

history=autoencoder.fit(
                x_train,
                x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,x_test)
                )
autoencoder.summary()
# autoencoder.save('E:/keras_data/autoencoder/CNN_autocoder.h5')
decoded_imgs=autoencoder.predict(x_test)
n=10
plt.figure(figsize=(24,4))
for i in range(n):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #展示重建
    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b')
    plt.plot(epochs, val_loss, 'r')
    plt.title('Training and validation loss')
    plt.show()
# 训练的acc_loss图
plot_training(history)