```python
import pandas as pd
pd.read_csv ("폴더명1/폴더명2/csv파일제목.csv")
#read_csv 에 Shift+Tab 누르면 자세한 사항 볼수있음

```

```python
titanic_df = pd.read_csv('titanic/train.csv')
titanic_df.head() #맨 앞부터 일부 데이터만 추출
titanic_df.tail() #맨 뒤부터 일부 데이터 추출
#괄호 안에 숫자 넣으면 숫자만큼의 로우을 보여줌
display(titanic_df.head()) #이 방법이 원래 정석임
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 100)
#옵션 주는 방법 - set_option으로 로우 갯수, 칼럼 길이도 늘릴 수 있고, 칼럼 갯수도 가능

```

```python
dic1 = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
# 딕셔너리를 DataFrame으로 변환
data_df = pd.DataFrame(dic1)
# 새로운 컬럼명을 추가
data_df = pd.DataFrame(dic1, columns=["Name", "Year", "Gender", "Age"])
# 인덱스를 새로운 값으로 할당. 
data_df = pd.DataFrame(dic1, index=['one','two','three','four'])

```

```python
titanic_df.info()
#DataFrame내의 컬럼명, 데이터 타입, Null건수, 데이터 건수 정보를 제공
titanic_df.describe()
#데이터값들의 평균,표준편차,4분위 분포도
#단 숫자형만 보여줌

```

```python
titanic_df['Pclass'].value_counts()
#value_counts()에서는 동일한 개별 데이터 값이 몇건이 있는지
3    491
1    216
2    184
#제일 많은거부터 차례대로 보여줌
print(titanic_df['Embarked'].value_counts(dropna=False))
S      644
C      168
Q       77
NaN      2
Name: Embarked, dtype: int64
#결측치가 있다면 dropna = False를 사용하기
#사용 안한다면 결측치도 포함시켜서 계산함
#dataframe도 가능함
titanic_df[['Pclass', 'Embarked']].value_counts()
#갯수 순으로 정렬됨
```

```python
col_name1=['col1']
list1 = [1, 2, 3]
array1 = np.array(list1)
pd.DataFrame(list1, columns=col_name1)
1차원 리스트로 만든 DataFrame:
    col1
0     1
1     2
2     3
pd.DataFrame(array1, columns=col_name1)
1차원 ndarray로 만든 DataFrame:
    col1
0     1
1     2
2     3
#만일 columns = 안에 이름으로 넣고 싶다면 ['col1'] 이런식으로 작성해야함
#2차원 리스트도 동일하게 하면 됨
```

```python
#딕셔너리를 데이터 프레임으로 변환하기
dict = {'col1':[1, 11], 'col2':[2, 22], 'col3':[3, 33]}
df_dict = pd.DataFrame(dict)

# DataFrame을 ndarray로 변환 - 이게 중요 ★
array3 = df_dict.values

# DataFrame을 리스트로 변환
list3 = df_dict.values.tolist()

# DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
```

```python
#새로운 칼럼 만들때 이렇게 하기
titanic_df['Age_0']=0
#기존 칼럼들을 이용해서 새로운 칼럼 만들기
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
#axis를 이용해서 삭제 but titanic 데이터 자체에서 삭제 되지는 않음
titanic_df.drop('Age_0', axis=1 )
#inplace =True 로 만들어야 데이터에 삭제 됨
titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
#inplace가 디폴트 값이여도 같은 이름으로 반환할 시에 삭제됨
titanic_df = titanic_df.drop('Fare', axis=1, inplace=False)
```

```python
#인덱스 값을 다른 값으로 할당할 순 없음
#reset_index를 하면 index를 0,1,2 로 다시 반환시키고 index 칼럼을 생성시킴
titanic_df.reset_index(inplace=False)
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입과 shape:',type(value_counts), value_counts.shape)
### before reset_index ###
3    491
1    216
2    184
Name: Pclass, dtype: int64
value_counts 객체 변수 타입과 shape: <class 'pandas.core.series.Series'> (3,)

new_value_counts_01 = value_counts.reset_index(inplace=False)
print('### After reset_index ###')
print(new_value_counts_01)
print('new_value_counts_01 객체 변수 타입과 shape:',type(new_value_counts_01), new_value_counts_01.shape)
### After reset_index ###
   index  Pclass
0      3     491
1      1     216
2      2     184
new_value_counts_01 객체 변수 타입과 shape: <class 'pandas.core.frame.DataFrame'> (3, 2)

new_value_counts_02 = value_counts.reset_index(drop=True, inplace=False)
print('### After reset_index with drop ###')
print(new_value_counts_02)
print('new_value_counts_02 객체 변수 타입과 shape:',type(new_value_counts_02), new_value_counts_02.shape)
### After reset_index with drop ###
0    491
1    216
2    184
Name: Pclass, dtype: int64
new_value_counts_02 객체 변수 타입과 shape: <class 'pandas.core.series.Series'> (3,)
```

```python
# DataFrame의 rename()은 인자로 columns를 dictionary 형태로 받으면 '기존 컬럼명':'신규 컬럼명' 형태로 변환
new_value_counts_01 = titanic_df['Pclass'].value_counts().reset_index()
index	Pclass
0	3	491
1	1	216
2	2	184
new_value_counts_01.rename(columns={'index':'Pclass', 'Pclass':'Pclass_count'})
Pclass	Pclass_count
0	3	491
1	1	216
2	2	184
```

```python
iloc[행,열] -> 위치 기반 인덱싱
loc[행 명칭, 열 명칭] -> 명칭 기반 인덱싱
-> 명칭 기반이기에 없으면 오류가 뜸
```

```python
#여러개의 칼럼을 [] 안에 리스트로 넣을 시에 dataframe이 되어 나옴
데이터[['a','b']] #a와 b의 칼럼명을 가진 데이터 프레임이 생성
데이터[['a']] #a의 칼럼명을 가진 2차식이 생성됨
데이터['a'] #1차식 출력
```

```python
#iloc는 주로 맨 마지막 칼럼을 가져올 때 자주 사용함
데이터.iloc[:,-1]
데이터.iloc[:,:-1] #맨 마지막 칼럼 제외 시킨 모든 데이터를 출력하고 싶을 때
#iloc는 불린 인덱싱 안됨
#loc는 불린 인덱싱 됨
#불린 인덱싱을 할때 각자의 조건을 () 해야됨
#&-> and, |-> or

```

```python
sort_values(by=['칼럼명',['칼럼명']],ascending=False) #내림차순
#sort할시 by 사용한다는 점
#데이터 전처리시 groupby() 사용
#데이터 건수 알고 싶을 시 shape를 사용
```

```python
데이터_groupby[['칼럼명','칼럼명']] #여기에 .count() .sum() 사용하면 됨
titanic_df.groupby('Pclass')['Age'].agg([max, min])
agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)
#gruopby 사용법 알아두기
#같은 키 값 'age'의 max나 mean 둘 다 입력시 두번째꺼만 반환시킴
titanic_df.groupby(['Pclass']).agg(age_max=('Age', 'max'), age_mean=('Age', 'mean'), fare_mean=('Fare', 'mean'))
#agg를 사용해서 칼럼명을 정해준다면 같은 키 값으로 사용 가능(dict으로는 안됨)
#pd.NamedAgg(column='',aggfunc='')으로 사용하는 것도 동일함

```

```python
데이터.isna() #데이터 결측치 True로 반환
데이터.isna().sum() #결측치의 갯수를 알 수 있음, count로 하면 안됨
.fillna('바꾸고 싶은값') #결측치를 어떻게 바꿀지

```

```python
#nunique 는 몇건의 고유값을 가지고 있는 갯수를 알려줌
#replace는 원본 값을 다른 값으로 변경
데이터['칼럼명'].replace('원본값','바꾸고 싶은 값')
#replace로 널값 변경 가능
replace(np.nan,'변경값')
```

```python
#입력 값 여러개인 lambda 사용법
map(lambda x : 식, 리스트)
#apply(lambda x : 식) 으로 사용
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x <= 60 else 
                                                                                  'Elderly'))
#else를 여러 개 사용하는 방법으로 조건문 사용한다는 점
#아니면 def 함수를 만들어서 apply(lambda x : 함수(x)) 사용하면 됨

```
