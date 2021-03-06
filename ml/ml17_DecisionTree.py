# RF. 숲이 있으면 나무가 있다

# 결정트리 모델. 결정나무. DecisionTree 여러개가 모여있으면 앙상블

# from sklearn.tree import DecisionTreeClassifier

# 트리 구조 model에서 주요 parameter는 #1) max_depth : 깊이 #2) feature_importance : 컬럼별 중요도

# 사용법 : print(model.feature_importances_)

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

import pandas as pd

#1. 데이터

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.8, random_state=42)
#
# # print(x_train.shape)    # (455, 30)
#
# # print(x_test.shape)     # (114, 30)
#
# # print(y_train.shape)    # (455, )
#
# # print(y_test.shape)     # (114, )
#
#
#
# #2. 모델 구성
#
# model = DecisionTreeClassifier(max_depth=4)
#
# # max_depth의 최고점을 찾아라
#
#
#
# #3. 훈련
#
# model.fit(x_train, y_train)
#
#
#
# #4. 평가
#
# acc = model.score(x_test, y_test)
#
# print("acc: ", acc)
#
#
#
# print(model.feature_importances_)
#
# # 30개 출력(30개의 숫자 중에 큰 숫자가 가장 큰 영향을 주는 컬럼이다)
#
# # 대회 나가서 PCA를 먼저 돌리고 model.feature_importances_를 실행시키면 빠른 속도로 결정 가능
#
#
# print(x_train.shape)
# print(x_test.shape)

x_train=pd.concat(x_train[:,1:2],x_train[:,7:8],axis=1)