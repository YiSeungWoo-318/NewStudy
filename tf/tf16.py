import tensorflow as tf
import numpy as np
from keras.datasets import mnist

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D
from keras.callbacks import EarlyStopping

#1. 데이터

(x_train, y_train), (x_test, y_test) =  mnist.load_data()

# print(x_train.shape)        # (60000, 28, 28)

# print(x_test.shape)         # (10000, 28, 28)

# print(y_train.shape)        # (60000,)

# print(y_test.shape)         # (10000,)

#1-1. 데이터 전처리

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

# print(y_train.shape)        # (60000, 10)

# print(y_test.shape)         # (10000, 10)

x_train = x_train.reshape(-1, 28*28).astype('float32')/255

x_test = x_test.reshape(-1, 28*28).astype('float32')/255



#1-2. 데이터 슬라이싱

x_data = dataset[:, 0:-1]

y_data = dataset[:, [-1]]

# print(x_data.shape) # (8, 4)

# print(y_data.shape) # (8, 1)



#1-3. feed_dict에 feed 될 텐서를 위한 placeholder 설정
hypothesis=tf.nn.softmax(tf.matmul(x,w)+b)

w2=tf.Variable(tf.random_normal_initializer([100,50],))



x = tf.placeholder('float32', [None, 4])

y = tf.placeholder('float32', [None, 1])



#2. 회귀 모델 구성

w = tf.Variable(tf.random_normal([4,1], name='weight'))

b = tf.Variable(tf.random_normal([1], name='bias'))

h = tf.matmul(x,w)+b



#2-1. cost 손실함수 정의

cost = tf.reduce_mean(tf.square(h-y))



#2-2. 최적화 함수 정의

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)



#3. 훈련

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(2001):

        cost_val, h_val, _ = sess.run([cost, h, opt], feed_dict={x:x_data, y:y_data})

        if step%10==0:

            print(step, "cost: ", cost_val, "\n 예측값: \n", h_val)

