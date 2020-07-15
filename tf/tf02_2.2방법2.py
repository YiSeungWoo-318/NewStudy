import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# session을 통과하지 않았으므로 자료형 출력
print("node1 : ", node1, "node2 : ", node2)

print("node3 : ", node3)

# node1 :  tf.Tensor(3.0, shape=(), dtype=float32) node2 :  tf.Tensor(4.0, shape=(), dtype=float32)
# node3 :  tf.Tensor(7.0, shape=(), dtype=float32)
