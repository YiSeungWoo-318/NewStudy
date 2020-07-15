from keras.preprocessing.text import Tokenizer
import numpy as np


docs={"너무 재밋어요", "최고에요", " 참 잘 만든 영화에요", "추천하고 싶은 영화입니다", "한 번 더 보고 싶네요", "글쎄요","별로에요"
      ,"생각보다 지루해요","연기가 어색해요", "재미없어요","너무 재미없다","참 재밋네요"}
#긍정 부정
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token=Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x=token.texts_to_sequences(docs)
print(x)


from keras.preprocessing.sequence import  pad_sequences

pad_x=pad_sequences(x,padding='pre')

print(pad_x)
pad_x=pad_sequences(x,padding='post')
print(pad_x)
pad_x=pad_sequences(x,padding='post',values=1.0)


word_size=len(token.word_index)+1
print("전체 토큰 사이즈 :", word_size)

from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten

model=Sequential()
# model.add(Embedding(word_size,10,input_length=5))
# model.add(Embedding(250,10,input_length=5))
model.add(Embedding(25,10))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

acc=model.evaluate(pad_x,labels)[1]
print(acc)