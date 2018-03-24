# 使用自动编码器进行图像去噪
# 我们把训练样本用噪声污染，然后使解码器解码出干净的照片，
# 以获得去噪自动编码器。首先我们把原图片加入高斯噪声，然后把像素值clip到0~1
from keras.models import Model
from keras.layers import Dense,Input,Conv2D,UpSampling2D,MaxPooling2D
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

(x_train,_),(x_test,_)=mnist.load_data()
# nuber=10000
# x_train=x_train[0:nuber]
# x_test=x_test[0:5000]
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train=x_train.reshape(len(x_train),28,28,1)
x_test=x_test.reshape(len(x_test),28,28,1)

#加入噪声
noise_factor=0.5
x_train_noisy=x_train+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noisy=x_test+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)
x_train_noisy=np.clip(x_train_noisy,0.,1.)
x_test_noisy=np.clip(x_test_noisy,0.,1.)
print(x_train_noisy.shape)
print(x_test_noisy.shape)

input_image=Input(shape=(28,28,1))
x=Conv2D(32,(3,3),activation='relu',padding='same')(input_image)
x=MaxPooling2D(pool_size=(2,2),padding='same')(x)
x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
encoded=MaxPooling2D(pool_size=(2,2),padding='same')(x)

x=Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
x=UpSampling2D(size=(2,2))(x)
x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D(size=(2,2))(x)
decoded=Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

autoencoder=Model(inputs=input_image,outputs=decoded)
autoencoder.summary()
# 为这里是让解压后的图片和原图片做比较， loss使用的是binary_crossentropy。
autoencoder.compile(
    loss='binary_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
)
history=autoencoder.fit(
                x_train_noisy,
                x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy,x_test),
                callbacks=[TensorBoard(log_dir='autoencoder',write_graph=False)]
                )
decoded_imgs=autoencoder.predict(x_test_noisy)

plt.figure(figsize=(30, 6))  # 设置 figure大小
n = 10
for i in range(n):
    ax = plt.subplot(3, n, i + 1)  # m表示是图排成m行,n表示图排成n列,p是指你现在要把曲线画到figure中哪个图上
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()  # 只有黑白两色，没有中间的渐进色
    ax.get_xaxis().set_visible(False)  # X 轴不可见
    ax.get_yaxis().set_visible(False)  # y 轴不可见

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
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

