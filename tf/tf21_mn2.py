import tensorflow as tf
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

dataset=np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape)

y=dataset[5:]
print(y.shape)
y=y.reshape(1,5)
a=[]
for i in range(5):
    a.append(dataset[i:i+5])

x=np.array(a)

x=x.reshape(1,5,5)

print(x.shape)
print(y.shape)

from sklearn.model_selection import KFold

kfold = KFold(n_splits=3)



sequence_length = 5
input_dim = 5
output = 5
batch_size = 1

X = tf.compat.v1.placeholder(tf.float32,(None,sequence_length,input_dim)) # 3차원
Y = tf.compat.v1.placeholder(tf.int32,(None,sequence_length)) # 2차원

weights=tf.Variable(tf.random.normal([1,5]),name='weights')
bias=tf.Variable(tf.random.normal([5]), name='bias')

cell=tf.keras.layers.LSTMCell(output)
hypothesis,_states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

#2-1. cost 손실함수 정의
cost = tf.reduce_mean(tf.square(hypothesis-y))

#2-2. 최적화 함수 정의
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

#3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        cost_val, h_val, _ = sess.run([cost, hypothesis, opt], feed_dict={X:x, Y:y})
        if step%100==0:
            print(step, "cost: ", cost_val, "\n 예측값: \n", h_val)
#
#
#
#
