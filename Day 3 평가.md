```python
True, False 반환하는 값을 숫자 1,0 으로 반환하고자 할때는 .astype(int) 사용
타켓이 불균형할시 이진법으로 나누는 것은 너무 안좋음
```

```python
#오차 행렬
TN | FP
-------
FN | TP
from sklearn.metrics import confusion_matrix
array([[405,   0],
       [ 45,   0]])
```

```python
정밀도 = TP/(FP+TP)
재현율 = TP/(FN+TP)
from sklearn.metrics import accuracy_score, precision_score , recall_score
```

```python
pred_proba = lr_clf.predict_proba(X_test)
#predict_proba를 사용해서 어느 값이 1에 가까운지 확인할 수 있음
[0.44935225 0.55064775] -> 왼쪽은 0이 될 확률, 오른쪽은 1이 될 확률
1에 가까운 수를 반환함 => 그래서 1이 반환됨
```

```python
from sklearn.preprocessing import Binarizer
X = [[ 1, -1,  2],
     [ 2,  0,  0],
     [ 0,  1.1, 1.2]]
# threshold 기준값보다 같거나 작으면 0을, 크면 1을 반환
binarizer = Binarizer(threshold=1.1) #임계값
print(binarizer.fit_transform(X))
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]]
#임계값을 낮추면 재현율이 높아지고 정밀도는 낮아짐
```

```python
정밀도를 100% 만들려면 확실한 경우에만 Positive로 나머지는 모두 Negative로 예측
재현율을 100% 만들려면 모든 경우를 Positive로 하면 됨
적절한 값 f1_score
from sklearn.metrics import f1_score
```

```python
roc_curve(y_true,y_score)
y_true -> 실제 클래스 값
y_score -> predict_prob()의 반환 값에서 positive 컬럼 예측 확률
from sklearn.metrics import roc_curve
```

```python
# 수정된 get_clf_eval() 함수 
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
def precision_recall_curve_plot(y_test=None, pred_proba_c1=None):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
```
