# 왜 AI FrameWork 를 개발하려고 하는가 ?

AI FrameWork 를 개발하면서 기존의 FrameWork(keras, tensorflow, pytorch) 에 대한 지식 획득

머신 러닝, 딥 러닝에 대한 정확한 이해를 하고 있는지에 대한 검증의 일환으로 FrameWork 개발을 생각

통계, 확률 등의 이론적 내용을 실제 적용시켜보고 변화를 확인해보기 위해 (최적화, 메모리 관리, 병렬화 등등)

단발성 프로젝트가 아닌 지속적으로 update 하고 관리할 수 있는 프로젝트를 진행하고 싶었음

# AI FrameWork 의 주요 목적과 기능

AI FrameWork 의 주요 목적과 기능은 사용자가 인공지능 및 머신러닝 모델을 쉽게 개발, 학습, 배포할 수 있도록 지원하는 것, 이러한 프레임워크는 다양한 AI 작업을 자동화하고 최적화하여, 효율적으로 작업을 수행

## 주요 목적

### 1. 개발 용의성 증대 

- 복잡한 AI 모델을 쉽게 개발할 수 있도록 다양한 도구와 라이브러리 제공

### 2. 생산성 향상 

- 반복적이고 시간 소모적인 작업을 자동화하여 개발자의 생산성을 높인다.

### 3. 성능 최적화 

- 모델 학습 및 추론의 속도와 효율성을 극대화한다.

### 4. 유연성 

- 다양한 알고리즘과 하드웨어를 지원하여 광범위한 응용 분야에 적용할 수 있게 한다.

### 5. 협업 지원 

- 코드 공유, 모델 버전 관리, 공동 작업 등의 기능을 통해 팀 단위 협업을 지원한다.

## 주요 기능

### 1. 데이터 처리

- 데이터 로드 및 변환 : 다양한 데이터 포맷을 로드하고 변환하는 기능

- 전처리 파이프라인 : 데이터 정규화, 결측치 처리, 데이터 증강 등 전처리 과정을 자동화하는 파이프라인 구성

- 배치 처리 : 데이터셋을 배치 단위로 나누어 학습에 사용할 수 있도록 지원

### 2. 모델 정의

- 모델 구성 요소 : 레이어, 활성화 함수, 손실 함수 등의 빌딩 블록 제공

- 커스텀 모델 : 사용자가 직접 정의한 모델 구조를 쉽게 구현할 수 있는 유연성

- 모델 시각화 : 모델 구조를 시각적으로 표현하여 이해를 돕는 기능.

### 3. 학습 및 추론

- 학습 루프 : 모델 학습을 위한 기본 학습 루프 제공.

- 옵티마이저 : 다양한 최적화 알고리즘을 제공하여 학습 속도와 성능을 향상

- 검증 및 평가 : 학습 과정에서 모델의 성능을 평가하고 검증 데이터로 모델의 일반화 능력을 측정

- 추론 : 학습된 모델을 사용하여 새로운 데이터에 대해 예측 수행

### 4. 최적화

- 양자화 및 프루닝 : 모델의 경량화 및 속도 최적화를 위한 기술 제공

- 하이퍼파라미터 튜닝 : 자동으로 최적의 하이퍼파라미터를 찾는 기능

- 분산 학습 : 여러 GPT/TPU 를 사용하여 모델 학습을 분산 처리

### 5. 배포 및 운영

- 모델 저장 및 로드 : 학습된 모델을 파일로 젖아하고 필요 시 다시 로드할 수 있는 기능.

- 서빙 : 모델을 실시간으로 제공하는 서버 환경 구성

- 모니터링 : 모델의 성능 및 사용 현황을 모니터링하는 도구 제공

### 6. 사용자 인터페이스 

- 직관적 API : 사용자가 쉽게 이해하고 사용할 수 있는 직관적인 API 설계

- 문서화 및 튜토리얼 : 상세한 문서와 예제 코드, 튜토리얼을 제공하여 사용자의 학습 곡선을 낮춤

### 예시 프레임 워크

- TensorFlow 

- PyTorch

- Keras 

- MXNet

- ONNX

