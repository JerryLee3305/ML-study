통계적인 시각화 - 히스토그램, 바이올린 플롯, 산포도, 바차트, 분위, 상관히트맵

업무 분석 시각화 - 바차트, 라인, 파이, 영역, 산포도, 방사형, 버블, 깔때기

이 중 ‘히스토그램, 바이올린 플롯, 산포도, 바차트, 상관히트맵’ - seaborn

matplotlib은 통계, 업무 분석 모두 사용 (중요)

직관적이지 않은 api로 익숙해지기까지 오래걸림

Figure, Axes 이해가 무조건 필요

Figure - 그림을 그리기 위한 Canvas 역할, 그림판 크기 등을 조절

Axes - 실제 그림을 가지는 메소드, 그외 X,Y축 Title 등의 속성 설정

```python
import matplotlib.pyplot as plt
plt.plot([1,2,3],[2,4,6]) #기본으로 설정된 axes에서 axes.plot()함수 호출
plt.title('Hello World') #내부적으로 axes.set_title()함수
plt. show() #내부적으로 Figure.show() 호출
plt.figure(figsize = (10,4)) #figure 크기가 가로 10, 세로 4 객체 생성

```

```python
fig, ax = plt.subplots() #figure과 axes 같이 가져오기
fig,(ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))
#행 1줄, 열 2줄로 되어 있는데 figure과 axes를 같이 가져오고, axes는 두개
#입력값으로 x,y가 모두 같은 크기를 가져야만 그릴 수 있음
plt.plot(x,y, color = 'red', marker = 'o', linestyle = 'dashed', linewidth = 2, markersize = 12)
#줄 색은 빨강, 마커 모양은 동그라미, 선 스타일은 점선
```

```python
plt.xlabel('x축 이름')
plt.ylabel('y축 이름')
#x축, y축 레이블 이름 설정
plt.xticks(ticks = 숫자, rotation = 숫자) #x축 틱 값 나누기, x축 틱 값을 회전
plt.xlim(0,50) #x값을 0~50으로 제한
plt.legend() #범례 표시 이것을 사용하려면 plt.plot(label = '레이블') label을 해줘야 함
#여러개의 플롯을 하나의 axes 안에 넣을려면 그냥 plt.plot 여러개 쓰고 plt.show 쓰면 됨
#plt.bar 써서 바, 선 같이 표현 가능
```

차트의 유형과 시각화 정보 차원

- 히스토그램
- 바이올린 플롯
- 바 플롯
- 스캐터 플롯
- 라인 플롯

Seaborn - Matplotlib보다 쉽고 이쁘고 pandas와 연동 가능

axes level - 개별 axes가 plot에 대한 주도적 역할

figure level - FacetGrid 클래스에서 개별 axes 기반 plot 그릴 수 있는 기능 통제

장점 = 여러개 subplot에서 plot 매우 쉽게 생성, 축과 타이틀 자동 인지, 쉽게 시각화

단점 = 새로운 api 적응, 커스터마이징 어려움, 그래서 잘 안쓰임
