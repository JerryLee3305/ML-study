```python
회귀 트리 만들 때 rmse 반환하는 스코어 함수 만들기
def get_model_cv_prediction(model, X_data, y_target):
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring = 'neg_mean_squared_error', cv = 5)
    rmse_scores = np.sqrt(-1*neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print(np.round(avg_rmse,2))
```

```python
1. 회귀트리 Regressor 생성
lr_reg = LinearRegression()
rf_reg2 = DecisionTreeRegressor(max_depth=2)
2. 테스트용 데이터 셋 생성
X_test = np.arange(4.5, 8.5, 0.04).reshape(-1, 1) #4.5~8.5 사이 0.04 씩 100개의 데이터 셋
3. 데이터 시각화 위해 피처와 타겟과 생성
X_feature = bostonDF_sample['RM'].values.reshape(-1,1)
y_target = bostonDF_sample['PRICE'].values.reshape(-1,1)
4. fit 시킨 후 예측 돌리기
lr_reg.fit(X_feature, y_target)
rf_reg2.fit(X_feature, y_target)
pred_lr = lr_reg.predict(X_test)
pred_rf2 = rf_reg2.predict(X_test)
5. 산점도와 회귀 예측선 시각화
ax1.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE)
ax1.plot(X_test, pred_lr )
ax2.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE)
ax2.plot(X_test, pred_rf2)
```

```python
#bike sharing demand 데이터 실습
시계열 자료 datetime 형식으로 변환(문자->날짜형)
데이터.datetime.apply(pd.to_datetime)
데이터['year'] = 데이터.datatime.apply(lambda x : x.year)
#이런 식으로 year, month, day, hour 뽑아낼 수 있음
```

```python
#seaborn의 subplot 4X2 데이터 바그래프 시각화
fig, axs = plt.subplots(figsize = (16,8), ncols = 4, nrows = 2)
cat_features = ['year', 'month','season','weather','day','hour','holiday','workingday']
for i, feature in enumerate(cat_features):
    row = int(i/4)
    col = i%4
    sns.barplot(x=feature, y = 'count', data = bike_df, ax = axs[row][col])
#각 피처들 별로 count에 맞게 그래프를 그림
```

```python
#평가 함수 만들기 rmsle, rmse, mae
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) **2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

def rmse(y, pred):
    return np.sqrt(mean_squared_error(y,pred))

def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    mae_val = mean_absolute_error(y,pred)
    print('RMSLE : {0:.3f}, RMSE : {0:.3f}, MAE : {0:.3f}'.format(rmsle_val, rmse_val, mae_val))
```

```python
회귀를 사용할 시에 먼저 정규화 형태인지 확인하는게 중요함
y_target.hist() 보면서 정규분포 모양인지 보고 아니면 정규화 시키기
->로그 변환
y_log_transform = np.log1p(y_target) #이것으로 다시 학습 예측 진행
y_target_log = np.log1p(y_target)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size = 0.3, random_state = 0)
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

y_test_exp = np.expm1(y_test) #하고나서 다시 expm1해서 원상태로 돌려야됨
pred_exp = np.expm1(pred)
```

```python
#상관계수 중요도 시각화
coef = pd.Series(lr_reg.coef_, index = X_features.columns)
coef_sort = coef.sort_values(ascending = False)
sns.barplot(x = coef_sort.values, y=coef_sort.index)
```

```python
#원핫인코딩
X_features_ohe = pd.get_dummies(X_features, columns = ['year','month','day','hour','holiday','workingday','season','weather'])
#원핫인코딩한 데이터를 이용하여 학습 예측 이후 평가
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size = 0.3, random_state = 0)
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1 = False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1:
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    evaluate_regr(y_test, pred)
    
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha = 10)
lasso_reg = Lasso(alpha = 0.01)

for model in [lr_reg, ridge_reg, lasso_reg]:
    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1 = True)
```

```python
#회귀트리로 수행
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 랜덤 포레스트, GBM, XGBoost, LightGBM model 별로 평가 수행
rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)
xgb_reg = XGBRegressor(n_estimators=500)
lgbm_reg = LGBMRegressor(n_estimators=500)

for model in [rf_reg, gbm_reg, xgb_reg, lgbm_reg]:
    # XGBoost의 경우 DataFrame이 입력 될 경우 버전에 따라 오류 발생 가능. ndarray로 변환.
    get_model_predict(model,X_train.values, X_test.values, y_train.values, y_test.values,is_expm1=True)
```

```python
#house price advanced regression
#타겟값 분포도 확인 kde = True로 하면 선을 그려줌
sns.histplot(house_df['SalePrice'], kde = True)
plt.show()
#결측치 확인 후 많은 것들은 드랍시킴, 다른 숫자형은 mean값으로 대체
#문자열의 경우 원핫인코딩 하기 pd.get_dummies 하면 결측치 더미 값은 0이 됨
```

```python
#학습 예측 후 회귀평가
#회귀한 것 평가
from sklearn.model_selection import cross_val_score
rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target, scoring = 'neg_mean_squared_error', cv = 5))
#cross_val_score에서 scoring을 neg_mean_squared_error로 했을 때 본래 작은 것이 성능이 좋지만 큰 것을 좋다고 판단하므로 -1을 곱해줘야함
from sklearn.model_selection import GridSearchCV #GridSearchCV도 마찬가지
grid_model = GridSearchCV(model, param_grid=params, scoring = 'neg_mean_squared_error',cv=5)
grid_model.fit(X_features, y_target) #fit을 시켜준 다음 구해야한다는 점
rmse = np.sqrt(-1*grid_model.best_score_) #model.best_score_ 를 하면 좋은 값이 도출 됨
```

```python
#right skew(왼쪽으로 치우쳐진 그래프)된 경우 로그변환 사용
#left skew(오른쪽으로 치우쳐진 그래프)된 경우 exponential/power 변환 적용 -> 원본 제곱을 함
#right skew => mode< median< mean
#left skew => mean< median< mode
```

```python
#데이터 분포 왜곡도 확인 후 높은 왜곡도 피처 추출
from scipy.stats import skew #왜곡도 정도 알려줌
features_index = house_df.dtypes[house_df.dtypes != 'object'].index #숫자형 피처들만 가져옴
skew_features = house_df[features_index].apply(lambda x : skew(x)) #왜곡도 정보 보여줌
skew_features_top = skew_features[skew_features >1]
skew_features_top.sort_values(ascending = False) #1이상인 데이터가 왜곡도가 큰 것이기에 이것을 내림차순 정렬
#다시 원핫인코딩을 한 이후 학습 및 예측 후 rmse 평가
```

```python
#scatter plot으로 이상치 있는지 확인하기
plt.scatter(x= house_df_org['GrLivArea'], y= house_df_org['SalePrice'])
plt.show()
#이상치 제거 <-이때는 조건문 두개 이용해서 인덱스 구한 이후 drop(axis=0)로 하기
#제거한 이후 다시 학습 예측 후 평가
```

```python
#회귀트리 학습 예측
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#중요도에 따라 top20 내림차순
def get_top_features(model):
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index = X_features.columns)
    ftr_top20 = ftr_importances.sort_values(ascending = False)[:20]
    return ftr_top20
#내림차순 시각화
def visualize_ftr_importances(models):
    fig, axs = plt.subplots(figsize = (24,10),nrows = 1, ncols = 2)
    fig.tight_layout()
    for i_num, model in enumerate(models):
        ftr_top20 = get_top_features(model)
        axs[i_num].set_title(model.__class__.__name__+'Feature Importances', size = 25)
        sns.barplot(x=ftr_top20.values, y=ftr_top20.index, ax = axs[i_num])
models = [best_xgb, best_lgbm]
visualize_ftr_importances(models)
```

```python
#Ridge와 Lasso에 서로 가중치를 줘서 혼합으로 예측값 도출
pred = 0.4 * ridge_pred + 0.6 * lasso_pred #개별 모델 학습, 예측 먼저하기
np.sqrt(mean_squared_error(y_test,pred['key 값'])) #rmse값
```

```python
#회귀 모델을 스태킹 사용해서 예측하면 성능 좋음
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수. 
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds ):
    # 지정된 n_folds값으로 KFold 생성.
    kf = KFold(n_splits=n_folds, shuffle=False)
    #추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화 
    train_fold_pred = np.zeros((X_train_n.shape[0] ,1 ))
    test_pred = np.zeros((X_test_n.shape[0],n_folds))
    print(model.__class__.__name__ , ' model 시작 ')
    
    for folder_counter , (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        #입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 셋 추출 
        print('\t 폴드 세트: ',folder_counter,' 시작 ')
        X_tr = X_train_n[train_index] 
        y_tr = y_train_n[train_index] 
        X_te = X_train_n[valid_index]  
        
        #폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행.
        model.fit(X_tr , y_tr)       
        #폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장.
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1)
        #입력된 원본 테스트 데이터를 폴드 세트내 학습된 기반 모델에서 예측 후 데이터 저장. 
        test_pred[:, folder_counter] = model.predict(X_test_n)
            
    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성 
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1,1)    
    
    #train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred , test_pred_mean
```

```python
# get_stacking_base_datasets( )은 넘파이 ndarray를 인자로 사용하므로 DataFrame을 넘파이로 변환. 
X_train_n = X_train.values
X_test_n = X_test.values
y_train_n = y_train.values

# 각 개별 기반(Base)모델이 생성한 학습용/테스트용 데이터 반환. 
ridge_train, ridge_test = get_stacking_base_datasets(ridge_reg, X_train_n, y_train_n, X_test_n, 5)
lasso_train, lasso_test = get_stacking_base_datasets(lasso_reg, X_train_n, y_train_n, X_test_n, 5)
xgb_train, xgb_test = get_stacking_base_datasets(xgb_reg, X_train_n, y_train_n, X_test_n, 5)  
lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm_reg, X_train_n, y_train_n, X_test_n, 5)
```

```python
# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 Stacking 형태로 결합.  
Stack_final_X_train = np.concatenate((ridge_train, lasso_train, 
                                      xgb_train, lgbm_train), axis=1)
Stack_final_X_test = np.concatenate((ridge_test, lasso_test, 
                                     xgb_test, lgbm_test), axis=1)

# 최종 메타 모델은 라쏘 모델을 적용. 
meta_model_lasso = Lasso(alpha=0.0005)

#기반 모델의 예측값을 기반으로 새롭게 만들어진 학습 및 테스트용 데이터로 예측하고 RMSE 측정.
meta_model_lasso.fit(Stack_final_X_train, y_train)
final = meta_model_lasso.predict(Stack_final_X_test)
mse = mean_squared_error(y_test , final)
rmse = np.sqrt(mse)
print('스태킹 회귀 모델의 최종 RMSE 값은:', rmse)
```
