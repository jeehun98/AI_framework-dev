# 개발 내역, 순차적으로 계속 작성

## 1. 데이터 전처리 부분의 구현 - 08/21

- data_processing 의 클래스가 아닌 함수로 변형

- 데코레이터, 데이터 검증, 논리적 경로명

    - data_processing decorator 작성 : 일단 numpy 형태가 맞는지만 향후 기능이 추가되어야 함 - 08/21
 
    - dacorators 폴더 이동 - 08/21

## 2. 모델 모듈 개발

- init 파일의 작성 : 08/22

- moodel 클래스 정의 (model.py 파일의 작성 - Model 클래스 정의)

    - model 클래스 내용 확인 - keras

    - model 클래스 필수 구현 method 작성
    
        - model 클래스 에서의 layer 는 참조 역할만??, 단순 인터페이스로 작용

    - layer 기능 구현, layers 속성은 해당 정보를 읽기 전용으로 제공하는 역할 수행
        
        -src/layers - 실제 기능, 이것부터 개발을 해야겠다.

- 연산의 정의, 관리를 수행하는 Operation 클래스 해당 클래스가 왜 layer 클래스의 부모 클래스가 되는지 생각부터