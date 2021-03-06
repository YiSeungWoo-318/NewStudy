import numpy as np

from keras.datasets import cifar10

from keras.utils import np_utils

from keras.models import Sequential, Input, Model

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt



#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape)        # (50000, 32, 32, 3)

# print(x_test.shape)         # (10000, 32, 32, 3)

# print(y_train.shape)        # (50000, 1)

# print(y_test.shape)         # (10000, 1)



#1-1. 데이터 전처리

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)



# print(y_train.shape)        # (50000, 100)

# print(y_test.shape)         # (10000, 100)



#2. 모델구성
from keras.optimizers import Adam
model=Sequential()

model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),activation='relu'))

model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(32,kernel_size=3,padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['acc'])

modelpath = './model/cifar100/{epoch:02d}-{val_loss:.4f}.hdf5'       

checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

earlystopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlystopping, checkpoint, tb_hist])



# print(hist.history.keys())



#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test, batch_size=100)



print("loss: ", loss)

print("acc: ", acc)



loss = hist.history['loss']

acc = hist.history['acc']

val_loss = hist.history['val_loss']

val_acc = hist.history['val_acc']



# print('acc: \n', acc)

# print('val_loss: \n', val_loss)

# print('loss_acc: \n', loss_acc)



#5. 시각화

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)

plt.plot(loss, marker='.', c='red', label='loss')

plt.plot(val_loss, marker='.', c='blue', label='val_loss')

plt.grid()

plt.title('loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend()



plt.subplot(2,1,2)

plt.plot(acc, marker='.', c='red', label='acc')

plt.plot(val_acc, marker='.', c='blue', label='val_acc')

plt.grid()

plt.title('acc')

plt.ylabel('acc')

plt.xlabel('epoch')

plt.legend()

plt.show()



# 튜닝

# epochs=71, batch=100, 노드=1024,max2,drop0.3,64,96,max2,32,drop0.3,flat

#loss:  4.235457990169525

#acc:  0.1842000037431717



# epochs=78, batch=100, 노드=320,max2,drop0.3,64,96,max2,32,drop0.3,flat

#loss:  3.128042597770691

#acc:  0.29980000853538513



# epochs=, batch=100, 노드=32,max2,drop0.2,64,96,max2,drop0.1,flat

#loss:  4.998792147636413

#acc:  0.2345999926328659