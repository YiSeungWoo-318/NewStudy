#tf06_01.py
#lr을 수정해서 연습
#0.01 - > 0.1 / 0.001 / 1
#epoch이 2000번 적게 만들어라..


import tensorflow as tf

# 그래프 수준의 난수 시드 설정

tf.compat.v1.set_random_seed(777)




x_train=tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train=tf.compat.v1.placeholder(tf.float32, shape=[None])

# random_normal 정규분포로부터의 난수값 반환 ([1])=shape

w = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')

b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis=x_train*w+b

# sess = tf.Session()

# sess.run(tf.global_variables_initializer())

# print("w의 가중치: ", sess.run(w))

# w의 가중치:  [2.2086694]



h = x_train * w + b



# 케라스 컴파일 시 가장 중요한 것 loss=(cost)

cost = tf.reduce_mean(tf.square(h - y_train))

# (h-y)**2의 합 / 개수 == mse



# 케라스 컴파일 시 옵티마이저(경사하강법). minimize(cost)=cost가 가장 적을 때 구하라

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)



# with 문 안에 sess가 포함

with tf.compat.v1.Session() as sess:

    # 전체 변수들이 싹 초기화

    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):

        # _:공백(실행은 되지만 결과값 출력 안하겠다)

        _, cost_val, w_val, b_val = sess.run([train, cost, w, b],feed_dict={x_train:[1,2,3],y_train:[3,5,7]})

        if step%20==0:

            print(step, cost_val, w_val, b_val)


    print("예측:",sess.run(hypothesis,feed_dict={x_train:[4]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train: [5]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train: [6]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train: [7]}))
    print("예측:",sess.run(hypothesis, feed_dict={x_train: [8]}))