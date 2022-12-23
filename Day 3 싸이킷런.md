```python
피처 - 데이터의 일반 속성 (타켓값 제외 속성)
피처를 통해 타겟 값을 학습 시킴(정답 데이터)
학습 데이터와 테스트 데이터 분리 -> 학습 데이터 기반으로 모델 학습 -> 모델 이용해 예측 -> 실제 결괏값과 평가

```

```python
DecisionTreeClassifier -이것을 이용해서 결정트리 만든다고 생각하면 됨
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from import 사용하는 방법도 이해해두기
```

```python
load_iris() -데이터를 로딩하는 방법
iris.data - 피처만으로 이루어진 numpy 데이터 호출 [a,b,c,d],[a,b,c,d]
iris.target - 숫자로 이루어져있지만 target_names로 볼시에 이름을 볼 수 있음
feature_names - 칼럼 이름들

```

```python
주로 피쳐는 X 사용, 타겟은 y를 사용(관습적 사용)
학습용 데이터를 a=DecisionTreeClassifier에 a.fit을 시킴 -> 학습 수행
a.predict(피쳐 테스트 데이터) -> 예측 수행
from sklearn.metrics import accuracy_score ->정확도를 평가하는 툴
```

```python
사이킷런 주요 모듈
	-예제 데이터 세트(sklearn.datasets)
	-데이터 분리, 검증&파라미터 튜닝(sklean.model_selection)
	-데이터 전처리 필요 가공 기능(sklean.preprocessing)
	-알고리즘에 큰 영향 미치는 피처 우선순위대로 셀렉션 작업 수행(sklearn.feature_selection)
	-텍스트, 이미 데이터의 벡터화된 피처 추출(sklearn.feature_extraction)
	-차원 축소 관련 알고리즘(sklearn.decomposition)
	-분류, 회귀, 클러스터링, 페어와이즈 대한 성능 측정 방법(sklearn.metrics)
	-앙상블 알고리즘(sklearn.ensemble)
	-선형 회귀, 릿지, 라쏘 및 회귀 관련 알고리즘(sklearn.linear_model)
	-나이브 베이즈 알고리즘(sklearn.naive_bayes)
	-최근접 이웃 알고리즘(sklearn.neighbors)
	-서포트 벡터 머신 알고리즘(sklearn.svm)
	-의사 결정 트리 알고리즘(sklearn.tree)
	-피처 처리 등 변환과, 알고리즘 학습, 예측 등 유틸리티(sklearn.pipeline)
```

```python
#예측 정확도 보는 식
print('예측 정확도: {0:.4f}'.format(accuracy_score(테스트용 타겟 데이터,학습용 피처 예측 데이터)))

```

```python
from sklearn.model_selection import KFold
KFold를 4로 설정시 3/4을 학습시키고 1/4을 검증 데이터로 사용하여 4번 반복
KFold(n_splits=4) -이 방식으로 작성해야함
하지만 불균형 분포도 데이터일 시(2만개 데이터 중 원하는 데이터가 100개라면 KFold 부적합)
from sklearn.model_selection import StratifiedKFold 사용하기
```

```python
skfold = StratifiedKFold(n_splits=3) 사용해서 split 호출 시 반드시 레이블 데이터 셋도 추가 입력해야함
for train_index, test_index  in skfold.split(features, label):
이전에는 features만 넣었으면 되었지만 stratifiedKFold 사용시에는 피처와 레이블 둘 값 모두 필요!!

```

```python
교차 검증 방법 cross_val_score()
cross_val_score(classifier, 피처, 레이블, scoring='accuracy' or 'recall', 'precision', cv=폴드를 몇번 할지)
StratifiedKFold와 동일한 값이 나옴 그러나 cross_val_score()가 더 간단히 쓸 수 있음

```

```python
from sklearn.model_selection import GridSearchCV
GridSearchCV(classfier, param_grid=parameters(딕셔너리 형태로 만든거), cv=교차 검증 횟수, refit=True(학습을 시킨거에서 가장 좋은 거를 가지도록))
parameters = {'max_depth':[1, 2, 3], 'min_samples_split':[2,3]}
#안에는 무조건 리스트로 작성 하나여도 무조건 리스트로 쓰기
grid_dtree.predict(X_test) 이거와 grid_dtree.best_estimator_ 이것이 같음
predict를 안시켜줘도 이미 best_estimator_에 학습이 되어 있음(refit=True 이기 때문에)
```

```python
#데이터 전처리
-결손값처리
-데이터 인코딩(문자열->숫자값) -레이블 인코딩, 원핫 인코딩
-데이터 스케일링(다른 척도를 같은 척도로 변환)
-이상치 제거
-피처들을 추출 및 가공
```

```python
#레이블 인코딩
머신러닝으로 데이터 처리시 문자열 값을 숫자 형태로 반환 시켜야함
fit()과 transform()를 사용하여 변환시킴
but 숫자형으로 반환 시키기에 숫자의 크고 작음으로 인해서 알고리즘에 문제가 발생할 수 있음
-> 그래서 #원핫 인코딩 사용
#원핫 인코딩
고유 값에 해당하는 칼럼에만 1을 표시하고 나머지는 0으로 표시
[0,0,1,0,0,0] 이런식으로 사용
fit(), transform() 사용하지만 인자를 2차원 ndarray로 입력
toarray()를 적용해 Dense형태로 변환해야함 -> pd.get_dummies(DataFrame)을 이용
```

```python
from sklearn.preprocessing import LabelEncoder
문자형 데이터를 fit 하고 transform 해야함 -> 숫자형태로 변경 됨
encoder = LabelEncoder()
label = encoder.fit_transform(데이터) #두개를 한번에 쓸 수 있음
encoder.classes_ #인코딩 한 클래스 이름을 알 수 있음
encoder.inverse_transform #숫자를 원본 값인 문자로 바꿀 수 있음
```

```python
from sklearn.preprocessing import OneHotEncoder
items = np.array(items).reshape(-1, 1) #무조건 2차원으로 변환시켜야함
똑같이 fit, transform 해줌
-> .toarray() 를 사용하여 인코딩 시켜줌

#판다스로 짧게 사용하는 방법 get_dummies 사용하면 됨
df = pd.DataFrame({'item':['TV','냉장고','전자렌지','컴퓨터','선풍기','선풍기','믹서','믹서'] })
pd.get_dummies(df)
```

```python
#스케일링 정규화
데이터 프레임에서 mean 과 var를 보면 각각의 칼럼들이 나눠서 보여줌
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(데이터)
scaler.transform(데이터) #이것도 fit_transform(데이터)로 사용해도 됨
평균이 0 분산이 1로 반환시킴

from sklearn.preprocessing import MinMaxScaler
이것도 fit_transform(데이터) -> 최소 0 최대 1로 스케일링
but 주의해야할게 스케일 기준을 하나로 잡아야 되기에 
test 데이터에 fit을 시키게 된다면 기준이 달라짐 -> 학습 데이터만 fit을 시킬 것
그래서 fit_transform()을 사용하기 보다는 fit을 학습 데이터만 시킨 후 그것을 기반으로 테스트 데이터를 transform 시켜야한다는 점

```

```python
titanic_df.isnull().sum() #개별 칼럼들의 결측치의 갯수를 알 수 있음
titanic_df.isnull().sum().sum() #모든 칼럼들의 결측치 갯수
titanic_df.dtypes[titanic_df.dtypes == 'object'].index.tolist()
#인덱스 값을 리스트로 반환하려면 .tolist()를 사용하면 됨
.str[:1] #스트링 값의 첫번째 단어만 가지고 오고 싶을 때 (=.str[0] 해도 같은 값 나옴)
```

```python
#성별과 생존수에 관계를 알고 싶을 때는 groupby를 사용
titanic_df.groupby(['Sex','Survived'])['Survived'].count()
import seaborn as sns
sns.barplot(x='Sex', y = 'Survived', data=titanic_df) #barplot 그림 그리는 방법
#hue 를 사용하면 x 데이터에서 또 기준을 나눌 수 있음
```

```python
1. info 결측치 확인 (.isna().sum())
2. 결측치가 너무 많으면 drop 칼럼 버림, 필요없는것도 axis=1
3. fillna 로 평균이나 전값 다음 값 사용하거나 drop axis=0로 그 로우를 제거
4. for 문 사용해서 피처들 문자열로 된 값을 fit_transform 시킴
```

```python
from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5):
    # 폴드 세트를 5개인 KFold객체를 생성, 폴드 수만큼 예측결과 저장을 위한  리스트 객체 생성.
    kfold = KFold(n_splits=folds)
    scores = []
    
    # KFold 교차 검증 수행. 
    for iter_count , (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        # X_titanic_df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
        # Classifier 학습, 예측, 정확도 계산 
        clf.fit(X_train, y_train) 
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))     
    
    # 5개 fold에서의 평균 정확도 계산. 
    mean_score = np.mean(scores)
    print("평균 정확도: {0:.4f}".format(mean_score)) 
# exec_kfold 호출
exec_kfold(dt_clf , folds=5)
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf, X_titanic_df , y_titanic_df , cv=5)
for iter_count,accuracy in enumerate(scores):
    print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))

print("평균 정확도: {0:.4f}".format(np.mean(scores)))
```

```python
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],
             'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(X_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터 :', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

# GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행. 
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test , dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy))
```
