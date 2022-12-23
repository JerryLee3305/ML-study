```python
1. 데이터셋 불러오기 pd.read_csv()
2. 데이터의 shape 파악, head로 칼럼명들 한번 보기
3. 데이터.info() 로 갯수와 결측치 봐보기
4. 불린 인덱싱을 통해 관심 있는 데이터 갯수 세기 
#cust_df[cust_df['TARGET']==1].TARGET.count()
5. describe()로 평균이나 최소 최대치 보면서 이상값 존재 여부 확인
6. 이상값들이 존재시 .value_counts() 로 몇개 있는지 보기
7. 데이터.replace(이상값, 바꿀값, inplace = True)
8. 관심 없는 거 drop 시키기
9. 피처와 라벨 만들기
#X_features = cust_df.iloc[:,:-1], y_labels = cust_df.iloc[:,-1]
```

```python
10. train 데이터와 test 데이터 분리 
from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X_features,y_labels,test_size=0.2, random_state=0)
#X_tr,X_val,y_tr,y_val = train_test_split(X_train,y_train, test_size = 0.3, random_state=0)
#훈련 데이터를 또 분리시키기
11. XGBClassifier을 이용해서 모델 학습 수행
xgb_clf.fit(X_tr, y_tr, early_stopping_rounds=100, eval_metric='auc', eval_set=[(X_tr, y_tr), (X_val, y_val)])
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])
```

```python
12. xgboost 사용(early_stopping 이 안되기에 KFold를 사용하기)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
def objective_func(search_space):
    xgb_clf = XGBClassifier(n_estimators = 100, max_depth=int(search_space['max_depth']) #int형 변환해야한다는점!!
                           ,min_child_weight = int(search_space['min)child_weight']) #int형 변환 주의
                            ,colsample_bytree = search_space['colsample_bytree']
                            ,learning_rate = search_space['learning_rate']
                           )
    roc_auc_list = []
    kf = KFold(n_split=3)
    for tr_index, val_index in kf.split(X_train):
        X_tr, y_tr = X_train.iloc[tr_index], y_train.iloc[tr_index]
        X_val, y_val = X_train.iloc[val_index]. y_train.iloc[val_index]
        xgb_clf.fit(X_tr, y_tr, early_stopping_rounds = 30, eval_metric = 'auc', eval_set=[(X_tr,y_tr),(X_val,y_val)])
        roc_auc_score(y_val, xgb_clf.predict_proba(X_val)[:,1]) #fit과 roc_auc쓸때 검증용 데이터 사용해야한다는점
        roc_auc_list.append(score)
    return -1* np.mean(roc_auc_list) #최소값을 위한 입력값 얻기 위해 -1을 곱해줘야함!!!
```

```python
12.  hyperopt를 이용해서 fmin()함수를 이용해 반복하여 목적함수의 최소값 가지는 best 값 출력
14. best 값을 가지고 다시 평가(auc와 early stopping 사용) -> 시각화 출력
```

```python
-------------------------------------------------------------
```

```python
#피처 엔지니어링을 할 때 데이터를 가공을 위해서 dataframe을 변환해야함
1. dataframe 복사 후 칼럼 삭제 후 복사된 것 반환하는 함수 사용
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time',axis = 1, inplace = True)
    return df_copy
```

```python
2. 복사된 데이터를 가지고 train과 test 데이터 셋을 만드는 함수 사용
def get_train_test_dataset(df=None):
    df_copy = get_preprocessed_df(df)
    X_features = df_copy.iloc[:,:-1]
    y_target = df_copy.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state = 0, stratify = y_target)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
```

```python
3. 평가 함수를 사용해서 정확도, 정밀도, 재현율, F1, AUC 파악하기
def get_clf_eval(y_test, pred=None, pred_proba = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall - recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```

```python
4. train과 test 데이터 셋 함수
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba)
```

```python
5. LogisticRegression #집단 예측해 분류
6. LGMBClassifier 사용
lgbm_clf = LGBMClassifier(n_estimators = 100, num_leaves = 64, n_jobs=-1, boost_from_average=False) #boost_from_average=False여야만 불균형 분포를 이룰 때 유리한 결과를 가져옴
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test = y_test)
```

```python
기울어진 데이터 가공 위해서는 1. StandardScaler 나 2.log 변환을 하면 됨

```

```python
1-1. StandardScaler 사용하는 방법
from sklearn.preprocessing import StandardScaler
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1,1)) #2차원으로 변형시켜줘야되서 fit_transform 시켜줌
    df_copy.insert(0, 'Amount_Scaled',amount_n) #insert(넣을 칼럼, '칼럼 제목', 넣을 데이터)
    df_copy.drop(['Time','Amount'],axis=1, inplace=True)
    return df_copy

```

```python
1-2. 변환 후 로지스틱 회귀 및 LGBM 학습 예측 및 평가 #이전에 썼던 함수에 그대로 넣음
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
lr_clf = LogisticRegression(max_iter = 100)
get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)
lgbm_clf = LGBMClassifier(n_estimators = 100, num_leaves = 64, n_jobs = -1, boost_from_average = False)
get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)
```

```python
2-1. log 변환 (np.log1p()를 사용한다는 점)
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis = 1, inplace = True)
    return df_copy
2-2. 똑같이 로지스틱과 lgbm 함
```

```python
3. 이상치 데이터 찾기
def get_outlier(df=None, column = None, weight = 1.5):
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25) #사분위수로 해서 찾기
    quantile_75 = np.percentile(fraud.values, 75)
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr*weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    outlier_index = fraud[(fraud<lowest_val) | (fraud>highest_val)].index #사분위수 최소 최대보다 작거나 큰 값이 이상값
    return outlier_index
outlier_index = get_outlier(df = card_df, column = 'V14', weight = 1.5)
```

```python
3-2. 이상치 제거 후 로지스틱, lgbm 함수 이용해 비교
def get_preprocessed_df(df = None):
    df_copy = df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount_Scaled', amount_n)
    df_copy.drop(['Time','Amount'],axis = 1, inplace = True)
    outlier_index = get_outlier(df = df_copy, column = 'V14', weight = 1.5)
    df_copy.drop(outlier_index, axis = 0, inplace = True)
    return df_copy
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
get_model_train_eval(lr_clf, ftr_train = X_train, ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)
get_model_train_eval(lgbm_clf, ftr_train = X_train, ftr_test = X_test, tgt_train = y_train, tgt_test = y_test)
```

```python
#oversampling 하기
from imblearn.over_sampling import SMOTE
SMOTE.fit_resample(x학습데이터, y학습데이터) #무조건 학습 데이터만 시키기
LogisticRegression 해서 학습데이터와 테스트 데이터 모델링 해보기 - 정밀도가 매우 낮아짐
```
