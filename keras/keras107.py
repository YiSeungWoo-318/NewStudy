
from keras.optimizers import Adam, RMSprop, SGD, Adadelta,Adagrad, Nadam, Adamax
from keras.layers import Input, LSTM, Dropout, Dense
from keras.models import Model
import numpy as  np



# 2) 모델링 함수 바꾸기
def build_model(drop, optimizer, lr):
    inputs = Input(shape = (28, 28), name = 'input')
    x = LSTM(64, activation = 'relu', return_sequences = True, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = LSTM(32, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(16, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer(lr=lr), metrics = ['accuracy'],
                  loss = 'categorical_crossentropy')
    return model

 # 하이퍼파라미터 함수 바꾸기
def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    lr = [0.1, 0.01, 0.001, 0.005, 0.007]
    optimizers = [RMSprop, Adam, Adadelta]
    dropout = np.linspace(0.1, 0.5, 5)
    return {'batch_size': batches,
            'lr':lr,
            'optimizer': optimizers,
            'drop': dropout}