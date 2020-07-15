import numpy as np
x=np.array([1,2,3,4])
y=np.array([1,2,3,4])

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))

model.add(Dense(1))


from keras.optimizers import Adam,RMSprop,SGD,Adadelta,Adagrad,Nadam,Adamax
# optimizer = Adam(1r=0.001)
# optimizer = RMSprop(1r=0.001)
# optimizer = SGD(1r=0.001)
# optimizer = Adadelta1r=0.001)
# optimizer = Adagrad=0.001)
# optimizer = Nadam=0.001)
optimizer = Adamax(lr=0.001)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x,y)