```python
대표적 알고리즘
-Naive Bayes
-Logistic Regression
-Decision Tree
-Support Vector Machine
-Nearest Neighbor
-Ensemble
```

```python
Graphviz
max_depth (트리의 깊이) - 디폴트는 none-> 계속해서 노드를 가짐
max_features - 디폴트는 none -> 전체의 피처를 사용, auto로 설정시 sqrt(전체의 루트 갯수)
min_samples_split - 분할 할 때의 최소 샘플 수(기준 수 이상이면 분할을 함, 보다 작으면 분할을 안함)
min_samples_leaf - sample의 수를 보고 나눌지 말지 구분
max_leaf_nodes -
```

```python
from sklearn.tree import export_graphviz
export_graphviz(dt_clf, out_file='tree.dot', class_names = iris_data.target_names,
                feature_names = iris_data.feature_names, impurity=True, filled=True)
#export_graphviz(DecisionTreeClassifier한 결과값, out_file = 어떤 모양으로 할지)
import graphviz
with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
#out_file을 tree.dot으로 했으니 open 시키고 그것들 다른 곳에 저장시킴
#graphviz는 컴퓨터 내 pip install 했기 때문에 그 자체로 import 시켜서 .Source로 그래프를 그리면 됨

#자기꺼 제외하고 나머지가 모두 0이 되도록 계속 쪼갬
```

```python
#중요도 시각화
sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print('{0}:{1:.3f}'.format(name, value))
#여러개로 묶여있을 시 zip을 사용해서 출력 가능
```

```python
#2차원 시각화 , 피처 2개, 클래스는 3개로
from sklearn.datasets import make_classification #데이터셋 중에서 분류화 만드는 것이 있음
X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_classes=3, n_clusters_per_class=1,random_state=0)
#산점도 형태로 그리기
plt.scatter(X_features[:, 0], X_features[:, 1], marker='o', c=y_labels, s=25, cmap='rainbow', edgecolor='k')
#plt.scatter(x축, y축, marker=어떤 모양으로 할지, c=컬러를 나누는 기준)

```

```python
#sep='\s+' 쓰면 공백을 불러들임
def get_human_dataset( ):
    
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당.
    feature_name_df = pd.read_csv('./human_activity/features.txt',sep='\s+',
                        header=None,names=['column_index','column_name'])
    
    # 중복된 피처명을 수정하는 get_new_feature_name_df()를 이용, 신규 피처명 DataFrame생성. 
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # DataFrame에 피처명을 컬럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    # 학습 피처 데이터 셋과 테스트 피처 데이터을 DataFrame으로 로딩. 컬럼명은 feature_name 적용
    X_train = pd.read_csv('./human_activity/train/X_train.txt',sep='\s+', names=feature_name )
    X_test = pd.read_csv('./human_activity/test/X_test.txt',sep='\s+', names=feature_name)
    
    # 학습 레이블과 테스트 레이블 데이터을 DataFrame으로 로딩하고 컬럼명은 action으로 부여
    y_train = pd.read_csv('./human_activity/train/y_train.txt',sep='\s+',header=None,names=['action'])
    y_test = pd.read_csv('./human_activity/test/y_test.txt',sep='\s+',header=None,names=['action'])
    
    # 로드된 학습/테스트용 DataFrame을 모두 반환 
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()
```

```python
#중복된 피처 찾는 방법
print(feature_dup_df[feature_dup_df['column_index'] > 1].count())
#결정 트리 정확도
dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train , y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test , pred)
print('결정 트리 예측 정확도: {0:.4f}'.format(accuracy))

# DecisionTreeClassifier의 하이퍼 파라미터 추출
print('DecisionTreeClassifier 기본 하이퍼 파라미터:\n', dt_clf.get_params())
```

```python
#GridSearchCV로 정확도와 최적 파라미터 확인하는 방법
from sklearn.model_selection import GridSearchCV
#아무리 단일 값이라도 리스트 안에 넣기!!
params = {'max_depth' : [ 6, 8 ,10, 12, 16 ,20, 24],'min_samples_split': [16]}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring = 'accuracy', cv=5, verbose=1)
grid_cv.fit(X_train , y_train)
print('GridSearchCV 최고 평균 정확도 수치:{0:.4f}'.format(grid_cv.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터:', grid_cv.best_params_)
```

```python
#피처 중요도 순으로 정렬 (20개만)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index) #인덱스는 칼럼명임
plt.show()
```

```python
#앙상블
#하드 보팅 - 다수의 classifier 의 다수결로 class 결정
#소프트 보팅 - class 확률을 평균하여 결정 -> predict_proba() 메소드를 이용하여 각 클래스별 확률을 구함

```

```python
# 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기 
vo_clf = VotingClassifier( estimators=[('LR',lr_clf),('KNN',knn_clf)] , voting='soft' )

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    test_size=0.2 , random_state= 156)

# VotingClassifier 학습/예측/평가. 
vo_clf.fit(X_train , y_train)
pred = vo_clf.predict(X_test)
print('Voting 분류기 정확도: {0:.4f}'.format(accuracy_score(y_test , pred)))

```

```python
#RandomForestClassifier 파라미터
-n_estimators : 결정 트리 갯수, 디폴트는 100개
-max_features : 디폴트는 auto(=sqrt) 전체 피처가 아닌 sqrt 갯수만큼 사용
-max_depth, min_samples_leaf 등 과적합 개선에 사용
```

```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=8)
rf_clf.fit(X_train , y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test , pred)
print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy))
```

```python
#importances를 시각화 주로 이렇게 많이 사용하니까 알아두기
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns  )
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()
```

```python
#부스팅 - 잘못예측한 데이터나 학습 트리에 가중치를 부여해 오류를 개선해 나가면서 학습
-learning_rate : 학습률, 0~1, 디폴트는 0.1, 주로 0.01~0.2사용
-n_estimators :디폴트 100, 많을수록 성능 수준 좋아지긴 하지만 오래 걸림
-subsample : 데이터 샘플링 비율, 디폴트 1, 과적합 염려될시 1보다 작은 값 사용

```

```python
#XGBoost
-early_stopping_rounds : 더이상 비용평가 지표가 감소하지 않는 최대 반복횟수
-eval_metric : 반복 수행시 사용하는 비용평가 지표
-eval_set : 평가 수행하는 별도 검증 데이터 세트
```

```python
import xgboost as xgb
from xgboost import plot_importance
-학습 데이터 80% 테스트 데이터 20% 쪼갬
X_train, X_test, y_train, y_test=train_test_split(X_features, y_label, test_size=0.2, random_state=156 )
-학습 데이터를 학습데이터 90%, 검증용 데이터 10% 쪼갬(early stopping할때 쓰임)
X_tr, X_val, y_tr, y_val= train_test_split(X_train, y_train, test_size=0.1, random_state=156 )
```

```python
#학습과 예측 데이터 세트를 DMatrix로 변환
dtr = xgb.DMatrix(data=X_tr, label=y_tr) -학습용
dval = xgb.DMatrix(data=X_val, label=y_val) -검증용
dtest = xgb.DMatrix(data=X_test , label=y_test) -테스트용
```

```python
#파이썬 런으로 xgb 실행 방법 (이때는 params를 밖에다가 쓴다는점!)
params = { 'max_depth':3,
           'eta': 0.05, #learning-rate
           'objective':'binary:logistic',
           'eval_metric':'logloss'
        }
num_rounds = 400
# 평가 데이터 셋은 'eval'
eval_list = [(dval,'eval')] 

# 하이퍼 파라미터와 early stopping 파라미터를 train( ) 함수의 파라미터로 전달
xgb_model = xgb.train(params = params , dtrain=dtr , num_boost_round=num_rounds ,
                      early_stopping_rounds=50, evals=eval_list )
#126번째에서 eval-logloss가 제일 낮아서 early_stopping_rounds가 50이므로 126번째에서 50번 반복을 해서 더 작은 값이 있는 지 보고 없으면 stop

```

```python
pred_probs = xgb_model.predict(dtest)
print('predict( ) 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
print(np.round(pred_probs[:10],3))

# 예측 확률이 0.5 보다 크면 1 , 그렇지 않으면 0 으로 예측값 결정하여 List 객체인 preds에 저장 
preds = [ 1 if x > 0.5 else 0 for x in pred_probs ]
print('예측값 10개만 표시:',preds[:10])
```

```python
#Feature Importance 시각화
fig, ax = plt.subplots(figsize=(10, 12)) #칼럼명을 보기 위해
plot_importance(xgb_model, ax=ax) #그림을 그리는 것은 ax임
```

```python
#사이킷런으로 XGBoost실행
from xgboost import XGBClassifier

# Warning 메시지를 없애기 위해 eval_metric 값을 XGBClassifier 생성 인자로 입력. 미 입력해도 수행에 문제 없음.   
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, eval_metric='logloss')
xgb_wrapper.fit(X_train, y_train, verbose=True)
w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]
```

```python
#early_stopping_rounds 사용
from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3)
evals = [(X_tr, y_tr), (X_val, y_val)]
xgb_wrapper.fit(X_tr, y_tr, early_stopping_rounds=50, eval_metric="logloss", 
                eval_set=evals, verbose=True) #early_stopping은 fit할때 넣어줘야한다는점!

ws50_preds = xgb_wrapper.predict(X_test)
ws50_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]
```

```python
#lightgbm
from lightgbm import LGBMClassifier
evals = [(학습용 피처, 학습용 타겟),(검증용 피처, 검증용 타겟)] #학습용 피처, 타겟은 안써도 무방함
lgbm_wrapper = LGBMClassifier(n_estimators=400, learning_rate=0.05)
evals = [(X_tr, y_tr), (X_val, y_val)]
lgbm_wrapper.fit(X_tr, y_tr, early_stopping_rounds=50, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
```

```python
#함수 만드는거 잘 알아두기
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```

```python
#GridSearch의 하이퍼 파라미털 갯수가 많아지면 문제가 생김(너무 오래걸림) - 데이터 세트가 작을때좋음
#베이지안 최적화 패키지 -hyperopt, optuna

```

```python
from hyperopt import hp
#검색공간 설정
search_space = {'x': hp.quniform('x', -10, 10, 1),  'y': hp.quniform('y', -15, 15, 1) }
#quniform(시작범위, 끝범위, 간격)
#딕셔너리 형태라는 점 주의!
```

```python
from hyperopt import STATUS_OK
# 목적 함수를 생성. 입력 변수값과 입력 변수 검색 범위를 가지는 딕셔너리를 인자로 받고, 특정 값을 반환
def objective_func(search_space):
    x = search_space['x']
    y = search_space['y']
    retval = x**2 - 20*y #실제적으로는 이렇게 안됨, 유추라서 이렇게 적은거임
    return retval  #권장 양식 : return {'loss': retval, 'status':STATUS_OK}
```

```python
from hyperopt import fmin, tpe, Trials
#fmin : 최솟값 반환 최적 입력변수값
# 입력 결괏값을 저장한 Trials 객체값 생성.
trial_val = Trials()

# 목적 함수의 최솟값을 반환하는 최적 입력 변숫값을 5번의 입력값 시도(max_evals=5)로 찾아냄.
best_01 = fmin(fn=objective_func, space=search_space, algo=tpe.suggest, max_evals=5
               , trials=trial_val, rstate=np.random.default_rng(seed=0)
              )

# Trials 객체의 vals 속성에 {'입력변수명':개별 수행 시마다 입력된 값 리스트} 형태로 저장됨.
print(trial_val.vals)
```

```python
#hyperopt에서 하이퍼 파라미터 튜닝
# max_depth는 5에서 20까지 1간격으로, min_child_weight는 1에서 2까지 1간격으로
# colsample_bytree는 0.5에서 1사이, learning_rate는 0.01에서 0.2사이 정규 분포된 값으로 검색. 
xgb_search_space = {'max_depth': hp.quniform('max_depth', 5, 20, 1),
                    'min_child_weight': hp.quniform('min_child_weight', 1, 2, 1),
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
               }
#정수형 파라미터일 때 quniform 사용한다는 점
#정규분포 형태를 뽑아내고 싶을 때 uniform(0.01,0.2)라면 0.01에서0.2 사이의 정규분포에서 뽑는 것임

```

```python
# fmin()에서 입력된 search_space값으로 입력된 모든 값은 실수형임. 
# XGBClassifier의 정수형 하이퍼 파라미터는 정수형 변환을 해줘야 함!!! 
# 정확도는 높은 수록 더 좋은 수치임. -1* 정확도를 곱해서 큰 정확도 값일 수록 최소가 되도록 변환
def objective_func(search_space):
    # 수행 시간 절약을 위해 n_estimators는 100으로 축소
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=int(search_space['max_depth']),
                            min_child_weight=int(search_space['min_child_weight']),
                            learning_rate=search_space['learning_rate'],
                            colsample_bytree=search_space['colsample_bytree'], 
                            eval_metric='logloss')
    #교차검증
    accuracy = cross_val_score(xgb_clf, X_train, y_train, scoring='accuracy', cv=3)
        
    # accuracy는 cv=3 개수만큼의 정확도 결과를 가지므로 이를 평균해서 반환하되 -1을 곱해줌. 
    return {'loss':-1 * np.mean(accuracy), 'status': STATUS_OK}
#accuracy는 높을수록 좋은 것인데 낮은 값을 출력하기에 -1을 곱해서 높은 값을 출력할 수 있도록함
```

```python
from hyperopt import fmin, tpe, Trials

trial_val = Trials()
best = fmin(fn=objective_func,
            space=xgb_search_space,
            algo=tpe.suggest,
            max_evals=50, # 최대 반복 횟수를 지정합니다.
            trials=trial_val, rstate=np.random.default_rng(seed=9))
print('best:', best)
#-1을 곱했기 때문에 최소의 값을 출력함
```
