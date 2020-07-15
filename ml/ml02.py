from sklearn.svm import LinearSVC              # SVM(Support vector machine)

from sklearn.metrics import accuracy_score    # : 결정 경계(desicion boundart) 분류를 위한 기준 선을 정의하는 모델

#1. 데이터

x_data = [[0,0], [1,0], [0,1], [1,1]]       # and 연산

y_data = [0,0,0,1]

# and_|___0___1_____

#     |

#  0  |   0   0

#     |

#  1  |   0   1

#     |

print(type(y_data))

#2. 모델

model = LinearSVC()            # 사용 모델 명시





#3. 훈련

model.fit(x_data, y_data)



#4. 평가, 예측

x_test = [[0,0], [1,0], [0,1], [1,1]]

y_pred = model.predict(x_test)



# score = model.evaluate(예측)

acc = accuracy_score([0,0,0,1], y_pred)          # evaluate = score()

print(x_test, "의 예측 결과: ", y_pred)

print("acc = ", acc)