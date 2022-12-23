```python
회귀 - 여러 개의 독립변수(피처)와 종속변수(결정 값)간의 상관관계 모델링 -> 최적의 회귀계수 추측
RSS 최소화, 규제를 적용하지 않은 모델

#최적의 회귀 모델은 전체 데이터 잔차 합이 최소가 되는 모델, 오류 값 합이 최소 될수 있는 최적의 회귀 계수 찾음
```

```python
LinearRegression(fit_intercept = 절편 넣을지 말지(True면 y0의 절편을 넣음, 이게 디폴트 값))
#선형 회귀의 경우 독립성 영향 많이 받음 -> 상관관계 커져 분산이 커짐 (=다중공산성) -> 독립적인 중요 피처만 남기고 제거 또는 규제
#평가지표 MAE(실제 값 - 예측값 을 절댓값 변환해 평균), RMSE(오류의 제곱에 루트를 씌움), RMSLE(RMSE에 로그를 적용)
# R^2 (예측값 / 실제값 , 1에 가까울수록 좋음)
#MAE - metrics.mean_absolute_error(실제값, 예측값)
#MSE - metrics.mean_squared_error(실제값, 예측값)
#RMSE - np.sqrt(MSE)
#MSLE - metrics.mean_squared_log_error(실제값, 예측값)
#R^2 - metrics.r2_score
```

```python
seaborn을 이용해 산점도와 회귀직선 표현
sns.regplot(x=feature, y='PRICE', data = bostonDF, ax = axs[row][col])
#이 점은 마지막 데이터 시각화에서 더 배울 예정
```

```python
lr = LinearRegression()
절편값 = lr.intercept_
회귀 계수값 = lr.coef_
회귀 계수값 큰 순 정렬 위해 series로 만들어서 sort_values(ascending = False) 사용하기
#cross_val_score에서 scoring = 'neg_mean_squared_error' 를 쓴다면 모두 음수가 되기 때문에 rmse를 구하기 위해서는 모두 -1을 곱해주고 sqrt 해주면 됨
neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

#값이 크거나 작은 것이 있다면 오버피팅일 가능성 있음
```

```python
#다항 회귀를 사용하려면 PolynomialFeatures 클래스로 다항 피처들로 변환한 것에 LinearRegression 적용
#일반적으로 PipeLine 클래스 이용해서 둘을 결합하여 다항 회귀 구현
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
```

```python
p_model = Pipeline([('poly', PolynomialFeatures(degree = 3, include_bias=False)), ('linear', LinearRegression())])
#degree = 1 이면 회귀직선임(선형회귀)
#degree가 높아질수록 overfitting 이 되기에 degree 값 설정에 유의해야함
fit을 시킨 후 predict하면 됨
mse와 rmse, r2_score를 보게 되면 많이 커지고 작아진 것을 확인할 수 있음
```

```python
#릿지 회귀 L2규제
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=10)
#alpha 값이 커질수록 평균 RMSE가 더 작아지긴 함
#alpha 값이 커지면서 오버피팅 된 것이 확 줄어들음
```

```python
#라쏘 회귀 L1 규제
#불필요한 회귀 계수를 급격히 감소시켜 0으로 만들고 제거함 -> 릿지랑 다름
#릿지는 0으로 가긴 해도 0으로 딱 떨어지지 않음

#엘라스틱넷 회귀는 L2,L1 규제 결합한 것
#라쏘회귀는 중요 피처만을 선택하고 나머지를 0으로 만드는 단점이 있기에 이를 완화하기 위해서 L2규제를 라쏘 회귀에 추가
#주요 파라미터로 alpha와 l1_ratio 있는데
alpha = L1 alpha + L2 alpha
l1_ratio = L1 alpha / (L1 alpha + L2 alpha)
```

```python
from sklearn.linear_model import Lasso, ElasticNet
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None, 
                        verbose=True, return_coeff=True):
    coeff_df = pd.DataFrame()
    if verbose : print('####### ', model_name , '#######')
    for param in params:
        if model_name =='Ridge': model = Ridge(alpha=param)
        elif model_name =='Lasso': model = Lasso(alpha=param)
        elif model_name =='ElasticNet': model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, 
                                             y_target_n, scoring="neg_mean_squared_error", cv = 5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print('alpha {0}일 때 5 폴드 세트의 평균 RMSE: {1:.3f} '.format(param, avg_rmse))
        # cross_val_score는 evaluation metric만 반환하므로 모델을 다시 학습하여 회귀 계수 추출
        
        model.fit(X_data_n , y_target_n)
        if return_coeff:
            # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가. 
            coeff = pd.Series(data=model.coef_ , index=X_data_n.columns )
            colname='alpha:'+str(param)
            coeff_df[colname] = coeff
    
    return coeff_df
```

```python
#선형 회귀 모델로 변환
1. 타겟 값 변환(target.hist() 로 정규 분포인지 보고 치우쳐져 있다면 로그변환)
2. 피처 값 변환
2-1. scaling적용 standard 변환 또는 MinMaxScaler 이용
2-2. 정규화 수행한 데이터에 PolynomialFeature 적용 변환 -> 과적합(오버피팅)유의
2-3. 심하게 기울어져있을경우 로그 변환(log1p)
```

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
# p_degree는 다향식 특성을 추가할 때 적용. p_degree는 2이상 부여하지 않음. 
def get_scaled_data(method='None', p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log':
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(degree=p_degree, 
                                         include_bias=False).fit_transform(scaled_data)
    
    return scaled_data
```

```python
#LogisticRegression
#주요 하이퍼 파라미터 - penalty, C, solver
penalty = 'l1', 'l2' 규제를 정할 수 있음
C = 1/alpha => C 값 작을 수록 규제 강도 큼
slover = 최적화 방식 디폴트 = lbfgs, 'liblinear'은 예전 디폴트 값인데 이게 속도가 좀 더 빠름
```
