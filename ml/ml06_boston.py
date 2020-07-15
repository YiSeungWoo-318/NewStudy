# # boston 회귀
#
#
#
# from sklearn.datasets import load_boston
#
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
#
# from sklearn.model_selection import train_test_split
#
# from sklearn.svm import SVC
#
# from sklearn.svm import LinearSVC
#
# from sklearn.metrics import accuracy_score, r2_score
#
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#
#
#
# #1. 데이터
#
# boston = load_boston()
#
# x = boston.data
#
# y = boston.target
#
# # print(y)
#
# # print(x.shape)        # (506, 13)
#
# # print(y.shape)      # (506,)
#
#
#
# #1-1. 데이터 전처리
#
# scaler = MinMaxScaler()
#
# x = scaler.fit_transform(x)
#
# # print(x[1])
#
# # print(x.shape)                  # (506, 13)
#
#
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)
#
#
#
# #2. 모델 구성
#
# # model = SVC()
#
# # ValueError: Unknown label type: 'continuous'
#
#
#
# # model = LinearSVC()
#
# # ValueError: Unknown label type: 'continuous'
#
#
#
# # model = RandomForestClassifier()
#
# # ValueError: Unknown label type: 'continuous'
#
#
#
# # model = RandomForestRegressor()
#
# # score:  0.8585668166937119
#
# # R2:  0.8585668166937119
#
#
#
# # model = KNeighborsClassifier()
#
# # ValueError: Unknown label type: 'continuous'
#
#
#
# model = KNeighborsRegressor()
#
# # score:  0.7510522283616072
#
# # R2:  0.7510522283616072
#
#
#
# #3. 컴파일, 훈련
#
# # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#
# hist = model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
#
#
#
# #4. 평가, 예측
#
# # loss, mse = model.evaluate(x_test, y_test, batch_size=1)
#
# score = model.score(x_test, y_test)
#
# # acc = accuracy_score(y_test, y_pred)
#
# r2 = r2_score(y_test, y_pred)
#
#
#
# print("score: ", score)
#
# # print("acc: ", acc)
#
# print("R2: ", r2)
#
#
#
# #------------------------------------------------------------------------------------------------------------------------
#

from sklearn.datasets import load_boston

from sklearn.svm import LinearSVC, SVC

from sklearn.metrics import accuracy_score, r2_score

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류, 회귀

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler



## 1. 데이터

data = load_boston()

x = data.data

y = data.target



x_train, x_test, y_train, y_test = train_test_split(

    x,y, random_state = 66, train_size = 0.8

)

# scaler = StandardScaler()

# scaler = MinMaxScaler()

scaler = MaxAbsScaler()

# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
#


## 2. 모델

ModelList = [KNeighborsClassifier(), KNeighborsRegressor(), LinearSVC(), SVC(), RandomForestClassifier(), RandomForestRegressor()]

Modelnames = ['KNeighborsClassifier', 'KNeighborsRegressor', 'LinearSVC', 'SVC', 'RandomForestClassifier', 'RandomForestRegressor']

for index, model in enumerate(ModelList):

    ## 3. 훈련

    try:

        model.fit(x_train, y_train)

    except ValueError:

        print("y값이 분류형 데이터가 아님!")

        continue

    ## 4.평가 예측



    y_pred = model.predict(x_test)


    score = model.score(x_test,y_test)



    print(Modelnames[index],'의 예측 score = ', score)

# RandomForestRegressor 의 예측 score =  0.9376427518921181




'''
modellist=[(),()...]
modelname=[(),()...]

for index, model in enumerate(modellist):
    try : 
    model.fit(x,y)
        :except Values error:
            print(":", )
            continue
            
            

'''
