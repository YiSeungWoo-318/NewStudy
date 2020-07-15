import tensorflow as tf

# 그래프 수준의 난수 시드 설정
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)


x_train=tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train=tf.compat.v1.placeholder(tf.float32, shape=[None])

# random_normal 정규분포로부터의 난수값 반환 ([1])=shape

w = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis=x_train*w+b

h = x_train * w + b

# 케라스 컴파일 시 가장 중요한 것 loss=(cost)

cost = tf.reduce_mean(tf.square(h - y_train))

# (h-y)**2의 합 / 개수 == mse

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

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