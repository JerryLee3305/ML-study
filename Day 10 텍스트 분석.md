NLP - 인간의 언어를 이해하고 해석하는데 더 중즘얼 두고 기술 발전 → 텍스트 분석도 더욱 정교하게 발전해짐

텍스트 분석 - 머신러닝, 언어 이해 등 활용해 모델 수립하고 정보를 추출해 비지니스 인텔리전스나 예측 분석등의 분석 작업 주로 수행

- 텍스트 분류
- 감성분석
- 텍스트 요약
- 텍스트 군집화와 유사도 측정

텍스트 문서 → Feature Vectorization(Bag of Words, Word2Vec) → ML학습

NLP와 텍스트 분석 패키지 - NLTK, Gensim, SpaCy

텍스트 전처리 - 텍스트 정규화

클렌징(불필요 문자, 기호 등 사전에 제거, html이나 xml 태그 및 기호 제거)

토큰화(단어 토큰화, n-gram)

필터링(불필요한 단어), 스톱워드 제거(분석에 큰 의미 없는 단어), 철자수정

Stemming/Lemmatization(어근 추출하여 기반 단어 원형을 찾아줌)

주로 NLP를 많이 사용하기에 이 코드들을 많이 사용할 일 없음

```python
from nltk import sent_tokenize #문장 토큰화
from nltk import word_tokenize #단어 토큰화
from nltk import ngrams #단어 토큰화 한 것을 묶음
nltk.download('stopwords') #stopwords 다운로드 받기
nltk.corpus.stopwords.words('english') #stopwords 사용하기
```

```python
from nltk.stem import LancasterStemmer #stemming, 'e'나 'y'같은 것이 빠질 가능성 높음
from nltk.stem import WordNetLemmatizer #lemmatization, 품사를 적어줘야함
#둘 다 어근을 보며 단어의 원형을 찾아줌
```

사이킷런 CountVectorizer 파라미터

max_df = 정수(정수 값 이하의 단어만 추출, 이상 값들은 필터링), 소수(상위 %는 필터링 시킴)

min_df, max_features

stop_words(english로 지정하면 영어의 스톱 워드로 지정된 단어 추출 제외)

ngram_range, token_pattern

```python
from sklearn.feature_extraction.text import CountVectorizer
객체화 시킨후 fit_transform 수행
print(객체화.vocabulary_) #피처 벡터화
```

희소행렬

coo형식 - 좌표 방식 의미, 0 아닌 데이터만 별도의 배열에 저장, 

csr 형식 - coo형식 중복적 문제 해결, 더 많이 사용

-행 위치 배열을 행 위치 배열의 인덱스를 구하고 +총 항목 개수 배열

형식 변환 하기 위해서는 Scipy의 coo_matrix(), csr_matrix() 함수 이용

```python
from scipy import sparse
sparse.coo_matrix(데이터)
sparse.csr_matrix(데이터)
```

텍스트 정규화 → 피처 벡터화 → 머신러닝 학습, 예측, 평가 → pipeline적용 → GridSearchCV 최적화

```python
#데이터셋에서 20개의 뉴스 그룹을 분류하기
from sklearn.datasets import fetch_20newsgroups
#subset에서 train, test, all 이 있기에 따로 train_test_split을 하지 않아도 됨
news_data = fetch_20newsgroups(subset='all',random_state=156)
#train과 test 데이터 따로해서 header와 footer, quote를 제거한 후 본문의 내용으로만 학습 수행

```

주의 : 피처 벡터화를 할 때 테스트 데이터에 fit을 시키면 안됨

마찬가지로 스케일링한 테스트 데이터에 대해서도 fit을 시키면 안됨

이유 : 기존 학습된 모델에서 가지는 피처의 갯수가 달라지기 때문임

```python
from sklearn.feature_extraction.text import CountVectorizer
cnt_vect = CountVectorizer()
X_train_cnt_vect = cnt_vect.fit_transform(X_train)
X_test_cnt_vect = cnt_vect.transform(X_test)
#만일 여기서 cnt_vect.fit_transform(X_test)를 사용하게 된다면 피처의 갯수가 달라짐
#위 train의 피처의 갯수와 같아야하기에 fit을 시키지 말기
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#로지스틱 회귀를 이용해 학습 예측 평가
lr_clf = LogisticRegression(solver = 'liblinear') #기존에는 'lbfgs'이지만 시간이 너무 오래걸리므로 liblinear를 사용
lr_clf.fit(X_train_cnt_vect, y_train) #학습
pred = lr_clf.predict(X_test_cnt_vect) #예측
print(accuracy_score(y_test, pred)) #평가
```

```python
#TF-IDF 피처 변환
from sklearn.feature_extraction.text import TfidVectorizer
tfidf_vect = TfidVectorizer()
tfidf_vect.fit(X_train)
X_train_tfidf_vect = tfidf_vect.transform(X_train)
X_test_tfidf_vect = tfidf_vect.transform(X_test) #이것도 마찬가지로 fit을 시켜주면 안됨

#위와 똑같이 로지스틱회귀로 학습 예측 평가하기
```

```python
#스탑워드 필터링 추가, 바이그램 늘리기(피처 수 늘어남), 갯수 300개 이상의 것 제거
tfidf_vect = TfidfVectorizer(stop_words = 'english',ngram_range =(1,2), max_df = 300)
```

로지스틱회귀 - ‘C’ 회귀계수의 규제를 위한 파라미터 ← alpha의 역수임

릿지와 라소의 경우 alpha(커질수록 규제가 강해짐)

C의 값이 커지면 커질수록 규제를 약하게 한다는 의미

```python
from sklearn.model_selection import GridSearchCV
# 최적 C 값 도출 튜닝 수행. CV는 3 Fold셋으로 설정. 
params = { 'C':[0.01, 0.1, 1, 5, 10]}
grid_cv_lr = GridSearchCV(lr_clf ,param_grid=params , cv=3 , scoring='accuracy' , verbose=1 )
grid_cv_lr.fit(X_train_tfidf_vect , y_train)
print('Logistic Regression best C parameter :',grid_cv_lr.best_params_ )

# 최적 C 값으로 학습된 grid_cv로 예측 수행하고 정확도 평가. 
pred = grid_cv_lr.predict(X_test_tfidf_vect)
print('TF-IDF Vectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))
```

사이킷런 파이프라인 사용 및 GridSearchCV와 결합

```python
from sklearn.pipeline import Pipeline
#파이프라인 안에 fit을 시킬 것들을 넣는다면 파이프라인이 전처리, 학습, 예측 모두 진행시켜줌
pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)),
    ('lr_clf', LogisticRegression(solver='liblinear', C=10))
])
#fit과 predict할 것을 그냥 넣기만 하면 끝
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(accuracy_score(y_test, pred))
```

```python
pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english')),
    ('lr_clf', LogisticRegression(solver='liblinear'))
])

# Pipeline에 기술된 각각의 객체 변수에 언더바(_)2개를 연달아 붙여 GridSearchCV에 사용될 
params = { 'tfidf_vect__ngram_range': [(1,1), (1,2), (1,3)],
           'tfidf_vect__max_df': [100, 300, 700],
           'lr_clf__C': [1,5,10]
}
#GridSearchCV(파이프라인, 하이퍼파라미터로 만든 params, cv, scoring)넣기
#똑같이 fit, predict, accuracy_score 진행
```

감성분석 - 문서의 주관적 감성, 의견, 감정, 기분 파악하는 방법

- 지도학습 기반의 분석
- 감성 어휘 사전 이용한 분석

```python
#영화 리뷰 실습
문자열 변경하는 방법
.str.replace(문자,변경문자)
import re
문자열만 가지고 싶다면 .apply(lambda x : re.sub('[^a-zA-Z]', ' ',x)

#리뷰를 가지고 sentiment와 학습 예측 평가하기(train_test_split)
#파이프라인으로 구축해서 벡터화와 로지스틱 회귀로 학습 예측 평가 하기
```

감성 어휘 사전 기반 감성분석 - SentiWordNet, VADER

비지도 학습용

SentiWordNet

- 문서를 문장 단위로 분해 → 문장을 단어 단위로 토큰화 하고 품사 태깅 → 단어 기반으로 synset과 senti_synset 객체 생성 → Senti_synset에서 긍정/부정 감성지수 구하고 합산해 특정 임계치 이상일 때 긍정 아닐 때 부정 결정

VADER (주로 사용)

- 소셜 미디어 감성 분석 용도
- SentimentIntensityAnalyzer 클래서 이용해 polarity_scores(문장) 이용
- compound 값이 0.1 이상이면 긍정, 이하면 부정감성

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti_analyzer.polarity_scores(데이터)
```

토픽 모델링

- 문서에 잠재되어 있는 공통 토픽(주제) 추출
- 구성을 보고 사람이 판별을 해줘야 됨
- LSA와 NMF - 행렬분해 기반 토픽 모델링
- pLSA와 LDA - 확률 기반 토픽 모델링

LDA - 문서별 단어 분포를 알고 있음(Document-Term 통해)

-베이즈 추론을 통해 문서별 토픽 분포와 토빅별 단어 분포를 알아야 됨

→ 베이즈 추론 사전 확률분포로 사용 되는 것이 디리클레 분포

1. count 기반 문서 - 단어 행렬 생성
2. 토픽의 개수(K)를 설정
3. 각 단어들을 임의의 주제로 최초 할당 후 문서별 토픽 분포와 토픽별 단어 분포 결정
4. 특정 단어 하나 추출 후 그 단어 제외 후 다시 재계산. 추출된 단어는 새롭게 토픽 할당 분포 계산
5. 다른 단어 추출 후 4번째 순서 반복, 모든 단어 토픽 할당 분포 재계산
6. 지정된 반복 횟수 만큼 계속 수행, 모든 단어 토픽 할당 분포 변경되지 않을 때까지 수행

초기화 파라미터 - n_components:토픽의 개수, doc_topic_prior : alpha(문서의 토픽 분포), topic_word_prior : 베타(토픽의 단어 분포), max_iter : 반복 횟수

```python
from sklearn.decomposition import LatentDirichletAllocation
20개 뉴스에서 카테고리 지정한 다음 all 가져오기
CountVectorizer를 통해 벡터화 -> fit_transform -> lda 객체 생성후에 fit
lda.components_ 에서 n_component 숫자 만큼 각 주제 별 단어가 나타난 횟수 정규화 - 숫자가 클수록 토픽에서 단어 차지 비중 높음
.argsort()[::-1] #가장 큰 값으로 정렬을 하는데 원래 가지고 있던 인덱스 값을 반환을 해서 표현
```

```python
lda 객체에 transform 하면 개별 문서별 토픽 분포 반환
데이터.filenames를 하면 저장된 경로와 함께 토픽 분류까지 나오므로
split('\\')하여 마지막 두개인 것만 반환 [-2:]
split을 시키면 , 로 반환 되니 join을 시킨후 빈 리스트에 append

토픽이 반환 되었으면 pandas에서 데이터프레임으로 만들어서
컬럼은 토픽 이름, 인덱스는 아까 만든 리스트, 데이터로 토픽들을 넣는다
그렇게 되면 각 단어가 어떤 토픽에 있는 지 알 수 있음
```

문서 군집화

텍스트 분류 기반 문서 분류는 사진에 결정 카테고리 값 가진 학습 데이터 셋 필요 but 문서 군지화는 학습 데이터 셋 필요 없는 비지도학습 기반!!!

```python
import glob, os #여러 파일이 있을 때 주로 사용함
path = r'경로복사' #r을 쓰게 되면 본래 \\ 써야할 것은 \ 복사 그 상태 그대로 사용해도 됨
all_file = glob.glob(os.path.join(path,'*.확장자'))
#glob.glob를 써서 경로에 있는 모든 파일들의 위치와 파일이름을 가져올 수 잇음
파일 이름을 경로에서 뒤에서 가져오고, 내용과 함께 데이터 프레임 형식으로 만듦
```

```python
from nltk.stem import WordNetLemmatizer
import nltk
import string
#ord 함수는 값을 유니코드 형식으로 변환 시켜줌 -> .이나 , 을 None으로 반환시키는 것임
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]
#토큰들에 대해서 어근변환(lemmatize)
def LemNormalize(text):
    return LemTokens(nltk, word_tokenize(text.lower().translate(remove_punct_dict)))
#우선 .이나 , 제거-> 소문자 변환 -> 단어 토큰화 -> 어근 변환
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
#토큰화를 위에 쓴 함수 LemNormalize 사용하고 stop_words, ngram_range, min_df, max_df 지정
feature_vect = tfidf_vect.fit_transform(document_df['opinion_text'])
#문서를 피처 벡터화 시킬 것이니 fit_transform 시켜준다.
from sklearn.cluster import KMeans #군집화 사용 위해 KMeans 사용
#KMeans 클러스터에 fit(피처벡터화) 한 이후 레이블과 좌표값(.cluster_centers_) 저장
기존 데이터에 레이블도 넣어주기
각 레이블 별로 어떻게 군집화 되어 있는지 확인하기
document_df[document_df['cluster_label']==1].sort_values(by='filename')
```

```python
군집별 핵심단어 추출하기
km_cluster.cluster_centers_ #이용하면 쉽게 알 수 있음
```

```python

문서 유사도 측정 - cosine Similarity
코사인 0도 = 1, 90도 = 0, 180도 = -1
피처 벡터 행렬은 count 기반이기에 음수값이 없음 => 코사인 유사도 0~1
1로 갈수록 유사한 지표 (크기보다는 벡터 방향성의 비교에 중점)
sklearn.metrics.pairwise.cosine_similarity(X,Y=None, dense_output=True)
(X,X)로 사용하면 쌍 형태로 각 문서끼리 코사인 유사도 행렬 반환

```
