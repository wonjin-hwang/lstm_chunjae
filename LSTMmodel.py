from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from datapreprocessing import make_dataframe
from data_scaling import data_scaling
import matplotlib.pyplot as plt

def LLSTM():
    train_x,train_y,test_x,test_y = data_scaling()
    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(90,5)))
    model.add(LSTM(units=128, activation='relu', return_sequences=True))
    model.add(LSTM(units=64, activation='relu',return_sequences=True))
    model.add(LSTM(units=32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # 모델 컴파일
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # 모델 학습
    model.fit(train_x, train_y, epochs=23, batch_size=64) # 23이 최적

    # 모델 평가
    loss = model.evaluate(test_x, test_y)
    print("테스트 데이터 손실:", loss)

    #원래 따로 빼려했는데 같이 둬도 괜찮을듯 model이나 test_x이런거 또 안불러와도 되서
    print("그림을 그려보자")
    pred_y = model.predict(test_x)

    plt.figure()
    plt.plot(test_y, color='red', label='real target y')
    plt.plot(pred_y, color='blue', label='predict y')
    plt.legend()
    plt.show()    