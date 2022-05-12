# 파이썬 패키지 가져오기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 하이퍼 파라미터
MY_EPOCH = 500
MY_BATCH = 64

# model = tf.global_variables_initializer();
heading = ['year', 'avgTemp', 'minTemp', 'maxTemp', 'rainFall', 'avgPrice']
raw = pd.read_csv('price data (1).csv')
raw = raw.drop('year', axis=1)
heading.pop(0)

print('원본 데이터 샘플 10개')
print(raw.head(10))

print('원본 데이터 통계')
print(raw.describe())

scalar = StandardScaler()
z_data = scalar.fit_transform(raw)


# numpy 에서 pandas로 전환
# header 정보 복구 필요
z_data = pd.DataFrame(z_data, columns=heading)

# 정규화 된 데이터 출력
print('정규화 된 데이터 샘플 10개')
print(z_data.head(10))

print('정규화 된 데이터 통계')
print(z_data.describe())

# 배추 데이터 사분할
# 학습용 입력값 || 학습용 출력값
# -------------------------
# 평가용 입력값 || 평가용 출력값

# 데이터를 입력과 출력으로 분리
print('\n 분리 전 데이터 모양: ',z_data.shape)
x_data = z_data.drop('avgPrice', axis=1)
y_data = z_data['avgPrice']

# 데이터를 학습용과 평가용으로 분리
x_train, x_test, y_train, y_test = \
    train_test_split(x_data,
                     y_data,
                     test_size=0.15)

print('\n학습용 입력 데이터 모양:', x_train.shape)
print('학습용 출력 데이터 모양:', y_train.shape)
print('평가용 입력 데이터 모양:', x_test.shape)
print('평가용 출력 데이터 모양:', y_test.shape)

# sns.set(font_scale=2)
# sns.boxplot(data=z_data, palette='dark')
# # plt.show()

########## 인공 신경망 구현 ##########

# 케라스 DNN 구현
model = Sequential()
input = x_train.shape[1]
model.add(Dense(200,
                input_dim=input,
                activation='relu'))
model.add(Dense(1000,
                activation='relu'))

model.add(Dense(1))

print('\nDNN 요약')
model.summary()

########## 인공 신경망 학습 ##########

# 최적화 함수와 손실 함수 지정
model.compile(optimizer='sgd',
              loss='mse')

print('\nDNN 학습 시작')
begin = time()

model.fit(x_train,
          y_train,
          epochs=MY_EPOCH,
          batch_size=MY_BATCH,
          verbose=0)
end = time()
print('총 학습시간 : {:.1f}초'.format(end - begin))

########## 인공 신경망 평가 및 활용 ##########


# 신경망 평가 및 손실값 계산
loss = model.evaluate(x_test,
                      y_test,
                      verbose=0)

print('\nDNN 평균 제곱 오차 (MSE): {:.2f}'.format(loss))


# 신경망 활용 및 산포도 출력
pred = model.predict(x_test)
sns.regplot(x=y_test, y=pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


#
# # 가설을 설정합니다.

#
# # 비용 함수를 설정합니다.
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# # 최적화 함수를 설정합니다.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
# train = optimizer.minimize(cost)
#
# # 세션을 생성합니다.
# sess = tf.Session()
#
# # 글로벌 변수를 초기화합니다.
# sess.run(tf.global_variables_initializer())
#
# # 학습을 수행합니다.
# for step in range(100001):
#     cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
#     if step % 500 == 0:
#         print("#", step, " 손실 비용: ", cost_)
#         print("- 배추 가격: ", hypo_[0])
#
# # 학습된 모델을 저장합니다.
# saver = tf.train.Saver()
# save_path = saver.save(sess, "./saved.cpkt")
# print('학습된 모델을 저장했습니다.')
