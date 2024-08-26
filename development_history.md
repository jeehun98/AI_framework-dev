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

        - Sequential layer 형태 기반, model 클래스 객체가 먼저 생성되는데, 이에 맞는 init 함수의 구현 - 08/23

        - model.add 메서드 구현

        - 여러 레이어가 모델 인스턴스에 쌓이는,

        - fit, predict 의 경우 model 의 부모 클래스, Trainer 에서 구현

    - layer 기능 구현, layers 속성은 해당 정보를 읽기 전용으로 제공하는 역할 수행
        
        - layer 층을 쌓는, 어떤 방법을 사용할 것인지 정의 - Sequential 형태 결정 : 08/23

        - src/layers - 실제 기능, 이것부터 개발을 해야겠다.

        - 레이어가 처음 생성되는 Input_layer, 여기에서만 구현되어야 하는 기능이 무엇일지

        - layer 가 쌓이면서 병행되는 연산이 무엇인지, 그것을 어디서 구현할 것인가 (interpreter 처럼 생각?)

        - layer 로 구현된 activation

        - init

        - build 함수 : 가중치 초기화 함수, 

        - call 함수 : 연산의 수행

        - get_config : 레이어의 구성 정보 반환

        - compute_coutput_shape : 출력 shape 계산

        - 중요한 점으로 어떤 차이를 두고 만들 것인지에 대한 고민을 해야 하는데 따라치지말고 이를 계속 생각해보자

    - Dense : 사용될 수 있는 parameter 에 대한 구현
    
        - activation - parameter

            - ops 내에서 해당 연산을 실제 수행, 일단 python 코드 내에서 구현하는 걸로 해보자

        - quantization_mode??

- 연산의 정의, 관리를 수행하는 Operation 클래스 해당 클래스가 왜 layer 클래스의 부모 클래스가 되는지 생각부터