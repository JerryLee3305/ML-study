```python
#군집화 알고리즘
K-Means, Mean Shift, Gaussian Mixture Model, DBSCAN(데이터 밀도에 따라)
```

```python
#K-Means Clustering
제일 많이 사용하는 알고리즘
#군집 중심점(centroid) 기반 클러스터링
군집 중심정을 설정 -> 각 데이터에서 가장 가까운 중심점에 소속 -> 중심점이 할당된 데이터 평균 중심점으로 이동
-> 각 데이터들 또 한번 가장 가까운 중심점에 소속 -> 중심점에 할당된 데이터들 평균 중심점으로 이동 -> 무한 반복 -> 더이상 소속 변화 없을 시 군집화 완료

장점 - 일반적 가장 많이 활용, 쉽고 간결, 대용량 데이터에도 활용 가능
단점 - 속성 개수가 많을 경우 정확도 떨어짐(차원 축소 필요할 수 있음), 반복 많을 시 수행 시간 길어짐, 이상치에 취약함

주요 파라미터 : n_clusters - 군집 중심점 개수
max_iter - 최대 반복 횟수
init = 'k-menas++'  -초기의 군집 중심점을 랜덤으로 놓지 않고 근처에 배치한다
```

```python
#붓꽃 데이터로 군집화해보기
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, random_state = 0)
#중심점을 3개 만든다, 랜덤으로 초기 중심점을 만들지 않는다, 최대 반복 횟수는 300번
kmeans.fit(irisDF) #fit을 시켜줘야됨!

kmeans.fit_predit(irisDF) #레이블값 반환
kmeans.fit_transform(irisDF) #좌표값 반환

#타겟 컬럼을 만들어주고 cluster 컬럼을 만들어 준 다음 groupby를 사용해서 한번 봐보기
```

```python
#총 4개의 값('sepal_length','sepal_width','petal_length','petal_width')을 2차원 평면에 나타내기 위해서 2차원 PCA 값으로 차원축소
pca = PCA(n_components = 2) #2차원으로 차원 축소
pca_transformed = pca.fit_transform(iris.data) #데이터를 fit_transform
irisDF['pca_x'] = pca_transformed[:,0]
irisDF['pca_y'] = pca_transformed[:,1]
#시각화에 사용할 데이터 x, y 구분해서 데이터프레임에 추가
```

```python
#시각화 하는 방법 1 - 마커를 이용해서 구분하기
marker0_ind = irisDF[irisDF['cluster'] ==0].index #cluster별로 인덱스 값을 구해서 나눈다
plt.scatter(x = irisDF.loc[marker0_ind,'pca_x'], y = irisDF.loc[marker0_ind,'pca_y'], marker = 'o')
#loc 기능을 사용해서 x, y 값들을 가져온다

#2 - 색상을 이용해서 구분하기
plt.scatter(x=irisDF.loc[:, 'pca_x'], y=irisDF.loc[:, 'pca_y'], c=irisDF['cluster'])
#마커 기준이 없기에 x,y 값들을 가져오고 color 기준을 cluster로 사용한다
```

```python
#clustering 알고리즘 테스트 데이터 생성
from sklearn.datasets import make_blobs #무작위 데이터들 생성
X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
#두개의 함수를 넣어줘야하고 n_samples(데이터 총 개수) 디폴트는 100
#n_features(데이터 피처 개수) - 시각화를 목표로 할 경우 주로 2개로 하여 첫번째는 x 두번째는 y로 사용
#centers(군집의 개수), cluster_std(군집 데이터 표준편차) - 만일 서로 다른 표준 편차를 가진 데이터 셋을 만들고 싶을 경우 [a,b,c] 값으로 넣어서 생성하면 됨

군집의 개수를 세개로 했으므로 y의 값도 0,1,2 값으로 나오는데 분포를 확인하고자 한다면 np.unique 사용하기
unique, counts = np.unique(y, return_counts=True)
#unique에는 y의 고유값들이 반환됨(0,1,2) return_counts=True를 쓰면 counts에 각 고유값들의 개수가 반환이 된다.
```

```python
clusterDF = pd.DataFrame(data = X, columns = ['ftr1','ftr2']) #X의 피처가 두개이므로 데이터프레임 형식을 만들 때 컬럼도 두개로 생성
clusterDF['target'] = y #타겟 값인 y도 넣어주기
#마커로 표현하는 데이터 시각화
target_list = np.unique(y) #np.unique를 사용할 때 고유값인 0,1,2만 반환함
markers = ['o','s','^']
for target in target_list:
    target_cluster = clusterDF[clusterDF['target']==target] #0,1,2인 데이터 가져옴
    plt.scatter(x=target_cluster['ftr1'], y=target_cluster['ftr2'],marker = markers[target], edgecolor = 'k')
	#마커별로 scatter plot 그리기
plt.show
```

```python
#군집 평가 - 실루엣 분석
각 군집 간 거리가 얼마나 떨어져 있는지, 동일 군집끼리 데이터가 얼머나 가까운지
#실루엣 계수
-1<=(bi - ai)/max(ai,bi)<=1
ai는 i번째 데이터에서 자신이 속한 군집 내 다른 데이터들 간의 거리 평균
bi는 i번째 데이터에서 가장 가까운 군집 내 다른 데이터 간의 거리 평균
1에 가까울 수록 다른 군집과 멀리 떨어져 있고, 0에 가까울 수록 다른 군집과 가깝다, -1에 가까울 수록 잘못 계산된 것
 
sklearn.metrics.silhouette_samples #각 실루엣 계수 반환
sklearn.metrics.silhouette_score # 전체 실루엣 계수 평균 반환, 많이 사용
score()가 1에 가까울수록 좋지만 높다고 해서 무조건 좋은 것은 아님
개별 군집의 평균값 편차가 크지 않아야하는 것이 중요
```

```python
from sklearn.metrics import silhouette_samples, silhouette_score
#모든 개별 데이터 실루엣 계수값
score_samples = silhouette_samples(iris.data, irisDF['cluster']) #피처 넣고, label을 넣어야됨
#모든 데이터 평균 실루엣 계수값
average_score = silhouette_score(iris.data, irisDF['cluster'])
#cluster 별로 평균을 구해보기
irisDF.groupby('cluster')['silhouette_coeff'].mean()
```

```python
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#이곳에서 클러스트별 시각화 코드 확인 가능
```

```python
#Mean Shift - KDE
데이터 분포가 높은 곳으로 동하면서 군집화를 수행
별도의 군집화 개수 지정할 필요 없음, 자동으로 개수 정함
커널 함수를 통해 확률밀도 함수 추정 방식
데이터 각각에 커널함수 적용한 값 모두 더한 뒤 건수로 나누어서 확률 밀도 함수 추정
bandwidth - 작으면 오버피팅, 크면 언더피팅
```

```python
#kde 시각화
https://seaborn.pydata.org/tutorial/distributions.html
#개별 관측 데이터에 대해 가우시안 커널 함수
from scipy import stats
bandwidth = 1.06*x.std()*x.size**(-1/5) #1.06*표준편차*크기^(-1/5)
support = np.linspace(-4,4,200)
stats.norm(x_i, bandwidth).pdf(support)#관측값, 표준편차
#pdf는 확률밀도함수 그려주는 것임
from scipy.integrate import trapz
density = np.sum(kernels, axis = 0)
density /= trapz(density, support)
plt.plot(support, density); #잘은 이해안가지만 그래프로 잘 그려줌

sns.distplot(x)#위에 plt.plot과 그래프가 같음

sns.kdeplot(x, shade = True) #위 그래프들과 같음
```

```python
#MeanShift
from sklearn.cluster import MeanShift
#최적의 bandwidth값 구하기
from sklearn.cluster import estimate_bandwidth
bandwidth = estimate_bandwidth(X,quantile=0.25) #quantile이 너무 작으면 시간이 오래걸림
```

```python
#GMM - K-Means는 길게 늘어진 데이터들은 잡을 수 없음
서로 다른 정규 분포로 결합된 원본 데이터 분포
#GMM 모수 추정 - 개별 정규 분포 평균과 분산 그리고 데이터가 특정 정규 분포에 해당할 확률 추정
-> EM(Expectation and Maximization)
-개별 데이터가 각각 정규 분포에 소속될 확률 구하고 가장 높은 확률 가진 정규 분포에 소속시킴(Expectation)
-정규분포에 소속되면 다시 해당 정규분포의 평균 분산 구함(Maximizaion)
-각 개별 데이터들이 소속 변경 되지 않을 때까지 계속 반복
```

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 3, random_state = 0).fit(iris.data)
#군집화 개수는 3개
gmm_cluster_labels = gmm.predict(iris.data) #predict는 label이 나온다는 점
irisDF['gmm_cluster'] = gmm_cluster_labels
```

```python
### 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 인자로 받아 클러스터링 결과를 시각화하는 함수  
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise=True
        else :
            cluster_legend = 'Cluster '+str(label)
        
        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70,\
                    edgecolor='k', marker=markers[label], label=cluster_legend)
        
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',\
                        edgecolor='k', marker='$%d$' % label)
    if isNoise:
        legend_loc='upper center'
    else: legend_loc='upper right'
    
    plt.legend(loc=legend_loc)
    plt.show()
```

```python
#DBSCAN - 데이터 밀도 차이를 기반, 밀도가 자주 변하거나 크게 변하지 않으면 성능 떨어짐
#피처 개수 많으면 성능 떨어짐
주요 파라미터
1. 입실론 주변 영역(epsilon) - 개별 데이터 중심으로 입실론 반경 원형 영역
2. 최소 데이터 개수(min points) - 개별 데이터 입실론 주변 영역 포함되는 데이터 개수

핵심포인트(core point) - 주변 영역 내 최소 데이터 개수 이상의 타 데이터 가질 경우
이웃포인트(neighbor point) - 주변 영역 내 위치한 타 데이터
경계포인트(border point) - 주변 영역 내 최소 데이터 개수 이상의 이웃 포인트를 가지고 있지 않아도 핵심 포인트를 이웃포인트로 가질 경우
잡음포인트(noise point) - 위 중 아무것도 아닌 데이터

eps : 입실론 주변 영역의 반경
min_samples : 입실론 주변 영역 내 포하뫼어야 할 데이터 개수(자신 포함된 개수임)
```

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 0.6, min_samples = 8, metric = 'euclidean')
dbscan_labels = dbscan.fit_predict(iris.data)
iris_result = irisDF.groupby('target')['dbscan_cluster'].value_counts()
#-1이 나온 값은 noise point임
#eps를 늘리면 noise 감소
#min_samples 증가하면 노이즈 증가함
```

online retail 실습

1. excel 파일 불러오기
2. quantity(수량)와 unitprice(개별 가격) 이 0인 것은 빼기
3. ID가 비어있는 것도 빼기
4. county가 가장 많이 있는 곳 확인 후 그곳만 사용하기
5. 새로운 컬럼인 ‘sale_amount’ 만들기 (quantity * unitprice)
6. ID를 정수형태로 변환시켜주기
7. invoiceno와 stockcode별 invoiceno의 평균 개수
8. 고객 기준으로 Recency(최근),  Frequency(빈도), Monetary(얼마 썼는지) cusd_df로 만들어서 표현하기
9. Recency를 2011.12.10 까지의 일수 차이로 표현하기 (datetime 이용, x.days+1 이용하여 Recency  정수로 변경)
10. 값 컬럼 요약하여 표시 #mean이 어디에 위치해 있는지 확인하기
11. KMeans로 군집화한 이후 실루엣 계수 평가 - 어떤 값들로만 할지, 스케일링하기, cluster는 3개로, 라벨 만들어서 실루엣 스코어 구하기
12. 시각화로 한번 봐보기
13. log변환을 통해 다시 만든 이후에 시각화보기

#실루엣 계수는 높을수록 좋은 것이지만 시각화로 봐본다면 로그변환한 것이 더 좋은 것임을 알 수 있음
