# 연령별 교통사고 중상율

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


연령 그룹화
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

연령대 별 중상자 비율 계산

<pre>
<code>
def count(x):
  return x[x!=1].sum()/x.count() #경상제외

age_severity = df.groupby('Age_Group')
</code>
</pre>
