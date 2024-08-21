각 프레임 워크의 data_processing 내용

# keras

1. 데이터 호출

2. 데이터 정규화 (별도의 함수 x)

3. 레이블 인코딩 (원-핫 인코딩, tocategorical)

4. 데이터 증강 (이미지 데이터 변환, ImageDataGenerator)

5. 배치 처리 (flow, flow_from_directory)

6. 데이터셋 분리 (train_test_split)

# pytorch

1. 전처리 작업 정의 ( transforms.Compose, Resize, ConterCrop, ToTensor, Normalize)

2. Dataset 클래스와 DataLoader 를 사용한 데이터 로드, 전처리

3. 사용자 정의 전처리 transform.Lambda

# scikitlearn

1. 결측값 처리, SimpleImputer

2. StandardScaler, MinMaxScaler, fit_transform

3. 인코딩, LabelEncoder, OneHotEncoder

4. 특징 선택, SelectKBest, Recursisve Feature Elimination

5. 데이터 분리, trian_test_split

6. 파이프 라인

# Tensorflow

1. tf.data.Dataset, 데이터 로드

2. 데이터 정규화, 임의 함수

3. 데이터 증강 Data Augmentation 

4. 데이터 셔플 

5. 배치 처리

6. 데이터 캐싱 및 Prefetching

# 필요 데이터 전처리 기능

1. 데이터 정제 - Data Cleaning

1.1 결측치 처리 : 데이터셋에 누락된 값이 있을 경우 이를 처리하는 기능

- 기능 : 데이터셋에 누락된 값이 있을 경우 이를 처리하는 기능

- 방법 : 삭제, 대체

1.2 중복 데이터 제거

- 기능 : 데이터셋에서 중복된 행을 식별, 제거

1.3 이상치 처리

- 기능 : 비정상적인 값의 처리

- 방법 : 이상치 식별, 제거,  대체

2. 데이터 변환 Data Transformation

2.1 데이터 정규화 및 표준화

- 기능 : 데이터의 범위를 일정한 값으로 조정

- 방법 : 정규화, 표준화

2.2 카테고리 데이터 인코딩

- 기능 : 범주형 데이터를 숫자형 데이터로 변환하는 기능

- 방법 : Label Encoding, One-Hot Encoding

2.3 텍스트 데이터 전처리

- 기능 : 텍스트 데이터를 모델이 처리할 수 있는 형태로 변환

- 방법 : 토큰화, 불용어 제거, 문자열 정규화

2.4 특징 공학 Feature Engineering

- 기능 : 원본 데이터에서 새로운 피처 생성

- 방법 : 피처 생성, 피처 선택

3. 데이터 셋 분할 Data Splitting

3.1 훈련, 검증, 테스트 데이터셋 분할

- 기능 : 데이터를 분할하여 일반화 능력을 평가

- 방법 : 분할, 교차 검증

4. 데이터 증강 Data Augmentation

4.1 이미지 증강

- 기능 : 이미지 데이터를 변환하여 데이터셋 확장

- 방법 : 회전, 반전, 크기 조절, 밝기 조절 등의 변환

4.2 텍스트 데이터 증강

- 기능 : 텍스트 데이터에 변형을 가하여 데이터셋 확장

- 방법 : 동의어 교체, 데이터 샘플링

5. 데이터 셋 분균형 처리 Handling Imbalanced Datasets

5.1 오버 샘플링과 언더 샘플링

- 기능 : 불균형한 데이터셋에서 클래스 균형을 맞추기 위한 기법

- 방법 : 오버샘플링, 언더 샘플링

6. 데이터 저장 및 로드 Data Saving and Loading

6.1 데이터셋 저장

- 기능 : 전처리된 데이터를 파일로 저장

6.2 데이터셋 로드

- 기능 : 저장된 데이터를 로드하여 재사용