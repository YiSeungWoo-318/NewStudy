import tensorflow as tf

hello = tf.constant("Hello AI")

print("hello 텐서의 자료형: ", hello)

#2.0이상버전에서는 따로 Session을 쓸 필요가 없다.
#hello 텐서의 자료형:  tf.Tensor(b'Hello AI', shape=(), dtype=string)