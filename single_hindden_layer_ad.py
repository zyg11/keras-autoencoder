from keras.models import Model
from keras.layers import Dense,Input
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

"""构建单隐藏层的自编码器
"""
# 32 floats -> compression of factor 24.5, assuming the input is 784 floats
encoding_dim=32
# this is our input placeholder
input_image=Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded=Dense(encoding_dim,activation='relu')(input_image)
# "decoded" is the lossy reconstruction of the input
decoded=Dense(784,activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder=Model(inputs=input_image,outputs=decoded)
#定义编码器
# this model maps an input to its encoded representation
encoder=Model(inputs=input_image,outputs=encoded)
#定义解码器
encoded_input=Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer=autoencoder.layers[-1]
decoder=Model(inputs=encoded_input,outputs=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

(x_train,_),(x_test,y_test)=mnist.load_data()
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
print(x_train.shape)
print(x_test.shape)
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
# 训练模型
history=autoencoder.fit(
    x_train,x_train,nb_epoch=100,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test,x_test)
)
#在测试集上进行编码和解码
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs=encoder.predict(x_test)
decoded_imgs=decoder.predict(encoded_imgs)
#显示结果
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    #display 原始
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #display reconstrustion
    ax=plt.subplot(2,n,n+i+1)
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