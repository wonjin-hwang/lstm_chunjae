import pandas as pd
import matplotlib.pyplot as plt
import holidays
import numpy as np
def make_dataframe():
    평년기온 = [-0.9, 1.4, 6.32, 12.24, 17.64, 21.76, 24.85, 25.56, 20.9, 14.64, 7.87, 1.11]
    평년습도 = [62.21, 60.58, 60.69, 60.9, 65.78, 73.22, 80.49, 79.35, 76.52, 70.95, 67.44, 63.93]
    # 월별 평균기온을 알면 월별로 얼마나추운지 알수있다
    # https://data.kma.go.kr/climate/average30Years/selectAverage30YearsList.do?pgmNo=113
    # 여기서 1991년~2020년 지역별월평균 다구할수있다.
    # 다 구하면 그거 다 평균내서 df에 평균기온 생성하고
    # 거기에 전부 평균값 떄려넣으면 될듯

    #하루마다의 습도나 온도를 구하기는 매우어려움(시간이없음) 그래서 걍 1달치로 ㅇㅇ


    df = pd.read_csv("한국가스공사_시간별 공급량_20181231.csv",encoding = "cp949")
    #df를 불러오니까 데이터타입이 이상하고 datetime도 아니길래 다 바꿔줬다. 아래가 그 과정
    df["연월일"]=pd.to_datetime(df["연월일"])
    df['시간'] = pd.to_timedelta(df['시간'], unit='h')
    df["날짜"]=df["연월일"]+df["시간"]
    df = df.drop(columns = ["연월일","시간"])
    column_to_move = "날짜"

    # 선택한 열의 위치를 맨 앞으로 이동
    df.insert(0, column_to_move, df.pop(column_to_move))


    #######################################################################################
    #날짜에서 00:00:00은 항상 애매하다
    #00:00:01로 바꿔버리자
    df['날짜'] += pd.to_timedelta(1, unit='S')
    #구분은 의미가 없다. 그냥 구분을 없애고 날짜별로 7개씩 중복이 있을테니 그거 전부합쳐버리자 왜냐고? 어차피 중요한건 공급량
    df = df.groupby("날짜").sum()
    df.reset_index(inplace = True)
    #월변수 하나 넣자 평균기온때문에 넣는거임
    df['월'] = df['날짜'].dt.month
    #월에 따른 평균기온을 넣어주자
    #월에 따른 평균습도를 넣어주
    df['평균기온'] = df['월'].map(dict(zip(range(1, 13), 평년기온)))
    df['평균습도'] = df['월'].map(dict(zip(range(1, 13), 평년습도)))

    #생각해보니까 우리의 목표는 2019년 0101부터 2019년 0401까지 예측하는거임
    #그니까 그냥 시간도 빼버리고 합쳐버리자
    df['날짜_날짜'] = df['날짜'].dt.date
    df=df.drop(columns = ["날짜"])
    df.rename(columns = {"날짜_날짜":"날짜"},inplace = True)
    #날짜(하루단위)로 그룹화 하고 공급량을 다 더해줬다.
    result = df.groupby("날짜", as_index=False).agg({'공급량(톤)': 'sum', '월': 'first', '평균기온': 'first', '평균습도': 'first'})
    # #datetime 으로바꿔
    result["날짜"]=pd.to_datetime(result["날짜"])
    # #공휴일이면 공휴일이라고 써놔
    import holidays
    holidays = holidays.KR()
    result['공휴일'] = result['날짜'].apply(lambda x: '휴일' if x in holidays else '근무')
    #공휴일 원핫인코딩
    one_hot_encoded = pd.get_dummies(result['공휴일'], prefix='휴일')

    # 기존 데이터프레임과 원-핫 인코딩된 열들을 합치기
    result = pd.concat([result, one_hot_encoded], axis=1)

    # "공휴일" 열 삭제
    result.drop(columns=['공휴일'], inplace=True)
    return result