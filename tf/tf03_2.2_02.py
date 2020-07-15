# 텐서플로우 1대 버전 사칙연산을 해보자
# 3+4+5
# 4-3
# 3*4
# 4/2
import tensorflow as tf

node0 = tf.constant(2.0)
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)

# tf.add_n은 많은 양의 텐서를 한번에 처리. []로 묶어줘야 처리 가능, tf.add = (x,y) 2가지만 처리

print("3+4+5 : ", tf.add_n([node1, node2, node3]))
print("4-3 : ", tf.subtract(node2, node1))
print("3*4 : ", tf.multiply(node1, node2))
print("4/2 : ", tf.divide(node2, node0))

# 3+4+5 :  tf.Tensor(12.0, shape=(), dtype=float32)
# 4-3 :  tf.Tensor(1.0, shape=(), dtype=float32)
# 3*4 :  tf.Tensor(12.0, shape=(), dtype=float32)
# 4/2 :  tf.Tensor(2.0, shape=(), dtype=float32)
#(tf.compat.v1.global_variables_initializer())도 2.2.0버젼에서는 필요없다.
