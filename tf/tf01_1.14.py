import tensorflow as tf

print("tf.__version__", tf.__version__) # tf.__version__ 2.2.0



hello = tf.constant("Hello AI")



print("hello 텐서의 자료형: ", hello)  # Tensor("Const:0", shape=(), dtype=string)

sess = tf.compat.v1.Session()

print("hello의 session 통과 후 출력: ", sess.run(hello))