```python
import numpy as np
list1 = [1,2,3]
array1 = np.array(list1)
array2 = np.array([[1,2,3],
                  [2,3,4]])
array3 = np.array([[1,2,3]])
#형태 파악
array1 array 형태: (3,)
array2 array 형태: (2, 3)
array3 array 형태: (1, 3)
```

```python
list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)
#변형 되긴 하지만 웬만하면 같은 형태를 사용하는 리스트를 넣기
[1. 2. 3.] float64
```

```python
array_float1 = np.array([1.1, 2.1, 3.1])
array_int2= array_float1.astype('int32')
print(array_int2, array_int2.dtype)
#타입을 변경할 수는 있음
[1 2 3] int32
```

```python
zero_array = np.zeros((3, 2), dtype='int32') #dtype를 써서 데이터 크기를 줄일 수 있음
print(zero_array.dtype, zero_array.shape)
[[0 0]
 [0 0]
 [0 0]]
int32 (3, 2)
#0으로 이루어진 ndarray를 생성: 만드는 이유는 형태를 우선적으로 만들기 위해
one_array = np.ones((3, 2))
print(one_array.dtype, one_array.shape)
[[1. 1.]
 [1. 1.]
 [1. 1.]]
float64 (3, 2)
```

```python
array1 = np.arange(10)
array2 = array1.reshape(2, 5)
print('array2:\n',array2)
#(2,5) 모양으로 변환
array2:
 [[0 1 2 3 4]
 [5 6 7 8 9]]
#(5,2) 모양으로 변환
array3 = array1.reshape(5,2)
print('array3:\n',array3)
array3:
 [[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
#갯수에 맞게끔만 가능
```

```python
array1 = np.arange(10)
array2 = array1.reshape(-1,5)
array2 shape: (2, 5)
[[0 1 2 3 4]
 [5 6 7 8 9]]
array3 = array1.reshape(5,-1)
array3 shape: (5, 2)
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
```

```python
array1 = np.arange(8)
array3d = array1.reshape((2,2,2)) #3차원 형태
array5 = array3d.reshape(-1, 1) #2차원으로 만들고 칼럼 갯수 1개로
array6 = array1.reshape(-1, 1) #2차원으로 만들고 칼럼 갯수 1개로
array1d = array3d.reshape(-1,) #3차원을 1차원으로
```

```python
np.sort(데이터)[::-1] #내림차순 정렬
array2d = np.array([[8, 12], 
                   [7, 1 ]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2d_axis1)
로우 방향으로 정렬:
 [[ 7  1]
 [ 8 12]]
컬럼 방향으로 정렬:
 [[ 8 12]
 [ 1  7]]
```

```python
np.argsort(데이터)
#행렬 정렬 시 원본 배열의 인덱스 반환
name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array= np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
name_array[sort_indices_asc]
#성적 오름차순 정렬로 원본 배열의 인덱스 출력 후 결과에 맞는 이름 출력
```

```python
np.dot(행렬1, 행렬2)
#행렬들의 곱

np.transpose(행렬)
#행렬의 전치
```
