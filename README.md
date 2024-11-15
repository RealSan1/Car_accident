# 교통사고 데이터 분석

데이터 (UK Car Accidents 2005-2015)
<pre>
<code>
import kagglehub

path = kagglehub.dataset_download("silicon99/dft-accident-data")
print("Path to dataset files:", path)
</code>
</pre>


전처리 과정 (주요 데이터 채집)
 <pre>
<code>
import pandas as pd

accidents = pd.read_csv('./Accidents0514.csv')
accidents.dropna(how='any', inplace=True)

casualites = pd.read_csv('./Casualties0514.csv')
casualites.dropna(how='any', inplace=True)

vehices = pd.read_csv('./Vehicles0514.csv')
vehices.dropna(how='any', inplace=True)

df = pd.DataFrame()

mergeData = pd.merge(pd.merge(accidents, casualites,on='Accident_Index'), vehices, on='Accident_Index')
df[['Accident_Index', 'Age_of_Driver', 'Accident_Severity', 'Weather_Conditions', 'Road_Surface_Conditions', 'Journey_Purpose_of_Driver', 'Day_of_Week', 'Time', 'Speed_limit']] = mergeData[['Accident_Index', 'Age_of_Driver', 'Accident_Severity', 'Weather_Conditions', 'Road_Surface_Conditions', 'Journey_Purpose_of_Driver', 'Day_of_Week', 'Time', 'Speed_limit']]
#인덱스, 운전자 연령, 사고 심각도, 날씨, 도로표면상태, 운전목적, 요일, 시간, 도로 속도 제한


df.to_csv('Drive.csv')
</code>
</pre>

Drive.csv

|Accident_Index|Age_of_Driver|Accident_Severity|''''|
|---|---|---|---|
|0|43|3||
|1|26|2|
|2|84|3|
|3|34|3|
|4|44|1|
|5|24|3|

## 연령별 교통사고 중상자 비율
### 유튜브에 [노인] 검색 시 교통사고 관련 검색어만 3개가 존재한다.
이처럼 연령이 증가할수록 교통사고 확률이 높아지는 걸까?

![유튜브 연관검색어](https://github.com/user-attachments/assets/4389ae9a-6f1e-4a50-9941-5a77a8782113)

### 시애틀 종단연구는 40년동안 6,000명의 정신적 기량을 연구
결과에서 운전 시 중요한 지각-반응속도가 나이가 들수록 급격하게 줄어드는 결과를 보이고 있다.

![연령별 인지능력](https://github.com/user-attachments/assets/264534c0-72a1-4431-8a02-f726a77f937b)


### 나이가 증가할수록 교통사고 시 환자의 중증도는 정말로 높아질까?

- 연령 그룹화
<pre>
<code>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Drive = pd.read_csv("./Drive.csv")
Drive.replace(-1, np.nan, inplace=True)
Drive.dropna(inplace=True)

df = Drive.loc[:, ['Age_of_Driver', 'Accident_Severity']]
df = df[df['Age_of_Driver'] > 17] # 17이상

bins = [17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100]
labels = ['20 under', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '70 over']

df['Age_Group'] = pd.cut(df['Age_of_Driver'], bins=bins, labels=labels, right=True) #연령 그룹화
df.drop('Age_of_Driver', axis=1, inplace=True)  
</code>
</pre>

Drive.csv

|Accident_Index|Age_of_Driver|Accident_Severity|''''|Age_Group|
|---|---|---|---|---
|0|43|3|''''|41-45|
|1|26|2|''''|26-30|
|2|84|3|''''|70 over|
|3|34|3|''''|31-35|
|4|44|1|''''|41-45|
|5|24|3|''''|21-25|

- 연령대 별 중상자 비율 계산

<pre>
<code>
def count(x):
  return x[x!=1].sum()/x.count() #경상제외

age_severity = df.groupby('Age_Group')
</code>
</pre>

- 그래프표현

<pre>
<code>
age_severity = df.groupby('Age_Group')
data = age_severity.agg(count)
data = data.reset_index()

plt.figure(figsize=(10, 6))
plt.plot(data['Age_Group'], data['Accident_Severity'], marker='o', linestyle='-', color='skyblue')

plt.xlabel('연령대')
plt.ylabel('중상자 비율')
plt.title('연령별 교통사고 중상자 비율')
plt.xticks(rotation=45)

plt.yscale('log')
plt.tight_layout()
plt.show()
</code>
</pre>

### 결과 

![image](https://github.com/user-attachments/assets/aa5a4d5f-9aee-445a-b4d8-ad83a635ca04)

연령대가 증가할수록 교통사고 증싱지 비율이 감소하는 추세이다.

### 결론
연령이 증가함에 따라 운전 솜씨가 향상되고, 면허 갱신 시기가 짧아지며 의사 소견서 제출이 요구된다. <br>
이러한 요인들로 인해 교통사고 발생 시 중증 환자의 비율이 감소한다.
[영국 노년 운전면허 갱신방법](https://www.gov.uk/renew-driving-licence-at-70#more-information)


## 가장 많은 교통사고가 발생하는 요일과 시간대

### 처리과정

<pre>
<code>
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./Drive.csv")
df.drop(['Age_of_Driver', 'Accident_Index', 'Unnamed: 0'], axis=1, inplace=True)

# 형식 변환
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
df['Time'] = df.groupby(pd.Grouper(key='Time', freq='1H'))['Time'].transform('first').dt.strftime('%H:%M')

day_mapping = {
    1: '일',
    2: '월',
    3: '화',
    4: '수',
    5: '목',
    6: '금',
    7: '토',
}
df['Day_of_Week'] = df['Day_of_Week'].map(day_mapping)


# 사고 발생 빈도를 계산 (시간대와 요일 기준)
heatmap_data = df.groupby(['Day_of_Week', 'Time']).size().unstack(fill_value=0)

day_order = ['일', '토', '금', '목', '수', '화', '월']
heatmap_data.index = pd.Categorical(heatmap_data.index, categories=day_order, ordered=True)
heatmap_data = heatmap_data.sort_index()

# 히트맵 그리기
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': '사고 발생 수'},
            annot_kws={'size': 8})

plt.title('사고 발생 시간대와 요일에 따른 빈도', fontsize=14)
plt.ylabel('요일', fontsize=12)
plt.xlabel('시간', fontsize=12)
plt.tight_layout()
plt.show()
</code>
</pre>

### 결과
![image](https://github.com/user-attachments/assets/93da2257-ea5d-4df6-a330-306b138e35e6)

### 결론
교통사고는 평일 출퇴근 시간대와 주말 11:00 ~ 14:00 시간대가 가장많이 발생한다.


