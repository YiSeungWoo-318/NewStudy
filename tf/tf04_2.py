
import tensorflow as tf

import numpy as np

@tf.function
def adder(a,b):
    return  a+b
print(adder(3,4.5))
#텐서플로우 1 computational graph 후 session으로 사용
#텐서플로우 2
#Session은ㄴ eager execution모드로 default가 됨
#@tf.function computational graph생성
#place holder 기능 없이 바로 넣으면 끝

#그리고 사칙연산은 tensorflow 내장 연산자 이용하자
'''
tf.add
tf.subtract
tf.multiply
divide
pow
negative (음수부호)
abs
sign (부호)
math.ceil
floor
math.square
math.sqrt
maximum
minimum
cumsum (누적합)
cumprod(누적곱)
'''


