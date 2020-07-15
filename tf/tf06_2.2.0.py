import tensorflow as tf


w = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

@tf.function
def hypothesis(x_train):
    return x_train*w+b

def h(x_train):
    return x_train*w+b

def cost(y_train):
    return tf.reduce_mean(tf.square(h(x_train=[1,2,3]) - y_train))


train = tf.optimizers.SGD(learning_rate=0.01)

for step in range(2001):

    # _:공백(실행은 되지만 결과값 출력 안하겠다)

    cost_val =  cost([3,5,7])

    if step % 20 == 0:
        print(cost_val)

print("예측:", hypothesis(4))
print("예측:", hypothesis(5))
print("예측:", hypothesis(6))
print("예측:", hypothesis(7))
print("예측:", hypothesis(8))

