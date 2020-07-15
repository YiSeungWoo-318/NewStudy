import tensorflow as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
idx2char=['e','h','i','l','o']


_data=np.array(['h','i','h','e','l','l','o'],dtype=np.str).reshape(-1,1)

print(_data.shape)
print(_data)
print(type(_data))

from sklearn.preprocessing import OneHotEncoder


enc=OneHotEncoder()
enc.fit(_data)
_data=enc.transform(_data).toarray()

print("=======")
print(_data)
print(type(_data))
print(_data.dtype)


x_data=_data[:6,]
y_data=_data[1:,]

print("==========")
print(x_data)
print("==========")
print(y_data)
print("==========")

y_data=np.argmax(y_data,axis=1)

print(y_data)
print(y_data.shape) #(6,)


x_data=x_data.reshape(1,6,5)
y_data=y_data.reshape(1,6)


print(x_data.shape)
print(y_data.shape)

sequence_lenth=6
input_dim=5


X=tf.compat.v1.placeholder(tf.float32(None,sequence_lenth,input_dim))
Y=tf.compat.v1.placeholder(tf.float32(None,sequence_lenth))


output=100
batch_size=6
print(X)
print(Y)


#.2모델구성


cell=tf.keras.layers.LSTMCell(output)
hypothesis,_states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

weights=tf.ones([batch_size,sequence_lenth])
sequence_loss=tf.contrib.seq2seq.sequence_loss(logits=hypothesis,targets=Y,weights=weights)
loss=tf.reduce_mean(sequence_loss)

train=tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction=tf.argmax(hypothesis,axis=2)


#훈련
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss=sess.run([loss,train],feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction,feed_dict={X:x_data})


        print(i,"loss:", loss, "prediction:",result, "true Y ", y_data)

        result_str=[idx2char[c]for c in np.squeeze(result)]
        print("\nPrediction str:",''.join(result_str))





