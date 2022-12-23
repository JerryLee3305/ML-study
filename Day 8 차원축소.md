```python
#PCA는 원본 데이터에서 가장 큰 변동성 기반으로 첫번째 축 생성
#두번째부터는 첫번째 축의 직각이 되는 축으로 축 생성 -> 축 개수 만큼 차원 축소
#원본 데이터의 공분산 행렬 추출 -> 고유벡터와 고유 값 분해 -> 고유 벡터로 선형 변환 -> PCA 변환 값 도출
#절차 1. 입력 데이터 셋 공분산 행렬 생성 2. 공분산 행렬의 고유벡터와 고유 값 계산 
#3. 고유 값 가장 큰 순으로 고유벡터 추출(PCA 변환 차수만큼) 4. 고유벡터를 이용해 새롭게 입력 데이터 변환
```

```python
#붓꽃 데이터 iris로 차원축소 해보기
```

```python
#target별 다른 모양으로 scatter plot 그려서 표현하기
markers=['^', 's', 'o'] #'^'는 세모, 's'는 네모, 'o'는 동그라미 마커
for i, marker in enumerate(markers):
    x_axis_data = irisDF[irisDF['target']==i]['sepal_length']
    y_axis_data = irisDF[irisDF['target']==i]['sepal_width']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])
plt.legend() #범례 표시
plt.xlabel('sepal length') #x레이블 이름
plt.ylabel('sepal width') #y레이블 이름
plt.show()
#각 i번째별로 maker를 그린다. for문으로 반복
```

```python
#PCA로 변환하기 이전에 원본 데이터를 정규 분포로 변환
iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:,:-1])
#fit과 transform 시키는 것을 같이 하는 fit_transform을 사용하기
#모든 피처들을 사용하는데 마지막은 target 값이기에 마지막 하나를 제외한 이후 스케일링하기
```

```python
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) #2차원으로 만든다는 것을 n_components = n을 사용해서 표현
iris_pca = pca.fit_transform(iris_scaled)
#스케일링 된 데이터를 다시 pca로 fit_transform을 시킴
#이 형태는 array 이므로 데이터 프레임 형식으로 바꾼 다음 target 값 추가해주기
#똑같이 마커별 for문 사용해서 시각화 그려보기
```

```python
print(pca.explained_variance_ratio_) #각 component별 변동성 비율 확인하기
#합치면 95%정도 되기에 두개의 축으로만 해도 잘 표현한다고 알 수 있음
#RandomForesClassifier를 사용해서 예측 성능 비교
rcf = RandomForestClassifier(random_state = 156)
scores = cross_val_score(rcf, iris.data, iris.target, scoring = 'accuracy',cv = 3) #원본 데이터
scores_pca = cross_val_score(rcf, pca_X,iris.target,scoring = 'accuracy', cv = 3)
```

```python
#신용카드 데이터 셋 PCA 변환
df = pd.read_excel('pca_credit_card.xls', header=1, sheet_name='Data').iloc[:,1:]
#엑셀 데이터 불러올 때는 pd.read_excel 사용
#header를 지정 안해주면 첫 행을 컬럼명으로 하기에 두번째 행을 컬럼명으로 하려면 header = 1로 설정
#엑셀 내 시트 명이 다를 경우 sheet_name으로 지정해주기
#빼고 싶은 데이터가 있을 경우 iloc를 사용해서 원하는 데이터만 가져오기
```

```python
#상관도 파악하기
피처.corr()를 사용하여 sns.heatmap(corr)를 넣으면 상관도 시각화 가능
#일부 속성만을 가지고 차원 축소 시키기
scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca = PCA(n_components = 2)
pca.fit(df_cols_scaled)
#마찬가지로 스케일링을 우선하고서 다시 pca로 fit 시키기
```

```python
#LDA(선형판별분석법)
#LDA 같은 클래스 데이터는 최대한 근접, 다른 클래스 데이터는 최대한 멀리
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#먼저 똑같이 스케일링 시킴
#CPA랑 다른게 fit을 시킬게 target 값을 넣어줘야함
iris_lda = lda.fit_transform(iris_scaled, iris.target)
#똑같이 시각화 그려보면 잘 분포된 것을 확인할 수 있음
```

```python
#SVD
#고유값 분해(행과 열 크기 같은 정방행렬만을 분해) 뿐만 아니라 행과 열 다른 행렬도 분해 가능
#대각선만 0이 아니고 나머지는 모두 0 =>대각행렬
#Truncated SVD 행렬 분해는 상위 r개만 추출하여 차원 축소
#잠재요인(Latent Factor)을 찾을 수 있어서 추천 엔진 및 잠재 의미 분석에 활용
from numpy.linalg import svd
```

```python
#Truncated SVD 행렬분해
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(iris_ftrs)
iris_tsvd = tsvd.transform(iris_ftrs)
# Scatter plot 2차원으로 TruncatedSVD 변환 된 데이터 표현. 품종은 색깔로 구분
plt.scatter(x=iris_tsvd[:,0], y= iris_tsvd[:,1], c= iris.target)

#스케일링 한 TruncatedSVD는 스케일링 한 PCA와 거의 유사한 산포를 가짐
```
