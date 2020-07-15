import tensorflow as tf
# print(tf.__version__) #2.20


tf.compat.v1.disable_eager_execution() #disable eager excution
# #1.14버전처럼 Session사용 가능


hello = tf.compat.v1.constant("Hello AI")

print("hello 텐서의 자료형: ", hello)

sess = tf.compat.v1.Session()

result=sess.run(hello)
print("hello의 session 통과 후 출력: ", result)