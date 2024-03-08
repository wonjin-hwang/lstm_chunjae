from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datapreprocessing import make_dataframe

def data_scaling():
    # 데이터 준비
    result = make_dataframe()
    dfx = result[["평균기온","평균습도","휴일_근무","휴일_휴일","공급량(톤)"]]
    for col in dfx.columns:
        scaler = MinMaxScaler()
        dfx[col] = scaler.fit_transform(dfx[[col]])
    dfy = dfx[["공급량(톤)"]]
    dfx = dfx[["공급량(톤)","평균기온","평균습도","휴일_근무","휴일_휴일"]]

    x = dfx.values.tolist()
    y = dfy.values.tolist()

    window_size = 90
    data_x, data_y = [], []
    for i in range(len(y) - window_size):
        try:
            _x = x[i:i+window_size]
            _y = y[i+window_size]
            data_x.append(_x)
            data_y.append(_y)
        except IndexError:
            # 인덱스 오류가 발생한 경우 건너뜁니다.
            continue

    train_size = int(len(data_y) * 0.8)

    train_x = np.array(data_x[0 : train_size])
    train_y = np.array(data_y[0 : train_size])

    test_size = len(data_y) - train_size
    test_x = np.array(data_x[train_size : len(data_x)])
    test_y = np.array(data_y[train_size : len(data_y)])


    print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
    print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)


    return train_x,train_y,test_x,test_y