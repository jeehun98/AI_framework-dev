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

            - 각 layer 인스턴스가 저장되어야 함

        - 여러 레이어가 모델 인스턴스에 쌓이는,

        - fit, predict 의 경우 model 의 부모 클래스, Trainer 에서 구현??

        - model.fit 에서 빌드 수행 - 08/28

            - build 의 역할이 가중치 초기화 하나라고 하면, fit 의 맨 처음 부분에서 build 를 수행해야 하겠네, model 클래스 내 is_build 속성 추가 - 08/28

            - sequential.build() 가 실행되면 각 레이어 방문, build 가 실행 가중치 값이 해당 레이어 인스턴스에 저장된다.

            - 모델의 상태를 저장하고, 복원 등의 기능을 수행하는 config 의 구현

            - model.fit 시 먼저 build의 수행

    - layer 기능 구현, layers 속성은 해당 정보를 읽기 전용으로 제공하는 역할 수행
        
        - layer 층을 쌓는, 어떤 방법을 사용할 것인지 정의 - Sequential 형태 결정 : 08/23

        - src/layers - 실제 기능, 이것부터 개발을 해야겠다.
        
            - layer 내 메서드, 속성의 regularizor 작성

        - 레이어가 처음 생성되는 Input_layer, 여기에서만 구현되어야 하는 기능이 무엇일지

            - Input_layer 의 구현, input_shape 의 정의, 등등 - 08/28

        - layer 가 쌓이면서 병행되는 연산이 무엇인지, 그것을 어디서 구현할 것인가 (interpreter 처럼 생각?)

            - 쌓이는 layer 구조에 대한 검증이 병행되어야 하는지, model.add 에서 실제 데이터가 들어오기 전에도 알 수 잇는??

        - layer 로 구현된 activation - activation method 와의 연동!!! 어떤 구존지 이해가 가버렷! 08/27

        - init

        - build 함수 : 가중치 초기화 함수, 

        - call 함수 : 실제 연산의 수행

        - get_config : 레이어의 구성 정보 반환

        - compute_coutput_shape : 출력 shape 계산

        - 결국 이런 모든 정보가 layer 클래스의 인스턴스에 저장되고, build, fit, summery 등 model 의 메서드에서 이러한 인스턴스 정보를 사용한다.

        - 중요한 점으로 어떤 차이를 두고 만들 것인지에 대한 고민을 해야 하는데 따라치지말고 이를 계속 생각해보자

    - Dense : 사용될 수 있는 parameter 에 대한 구현
    
        - activation - parameter

            - Layer 클래스를 상속받은 Dense 층에서 저장되는 속성값으로 activation, 

            - activations __init__ 파일의 get 을 통해 해당 함수를 저장할 수 있음

            - ops 내에서 해당 연산을 실제 수행, 일단 python 코드 내에서 구현하는 걸로 해보자

        - quantization_mode??

    - layers Layer 클래스를 상속받는 layer 들의 구현

        - activation 08/27

        - flatten 08/27

- 연산의 정의, 관리를 수행하는 Operation 클래스 해당 클래스가 왜 layer 클래스의 부모 클래스가 되는지 생각부터

## 3. 트레이너 개발

- 모델의 정의 후 model.fit 등의 연산이 수행되는 트레이너의 개발 시작 - 08/28

    - trainer

        - call 을 통해 각 레이어에 대한 연산들이 수행, 각 레이어들의 call 메서드 구현 필요 - 08/28

        - Sequential 클래스 내 call 메서드와 관련 함수, 속성의 정의 - 08/28

        - 각 layer 클래스 내 call 메서드 구현 (Dense) - 08/28

    - data_adapter_util


## 4. 계산 그래프 생성 및 c++ API 연동

- numpy 를 통한 파이썬 내부 계산이 아닌... 계산 그래프 생성 후 이러쿵 저러쿵 해야

    -

# 1. 다시 처음부터... 08/29

- 반환되는 객체의 타입을 명시적으로 지정해주는 typing.cast 의 사용, Sequential 클래스와 별도의 Layer 클래스에서도 명시해줘야겠다

    - Sequential 이 Layer 의 자식 클래스가 될 수 있는 것은 모델 자체가 Layer 가 될 수 있기 때문, 

    - Layer 의 하위 자식의 경우 init 메서드 내에서 추가적으로 구현해야 하는 정보 같은게 없어서 구현 X

- 모델의 정보를 저장할 get_config 메서드의 작성, 모델의 구조를 가지고 갈 수 있도록

    - model.get_config() 를 통해 해당 모델 layer 에 대한 정보를 호출할 수 있음, 더 필요한 정보가 뭐가 있을 지

    - model.get_compile_config() 구현 완료~! 야호 08/29

- 레이어가 쌓이는, model.add() 와 함께 input_shape, 가중치 초기화, built 지정의 설정

    - Dense 클래스의 build, input_shape 의 지정과 가중치 초기화

    - Layer 클래스의 build, build 상태 정보인, built 의 지정

        - 가중치 초기화와 input_shape 의 지정만 한다면, Flatten 은 초기화, 생성 시 built 로 지정할 수 있나?

        - input_shape 저장 확인 - 08/30

        - build_config 구현 : input_shape 의 저장

- 모델의 정보들이 담겨있는 config 파일들, 직렬화 하여 전달가능한 형태로 변경 완료 - 08/30

    - 입력 데이터도 전달 고고헛 (fit 에서 전달될 듯)

# 2. C++ 내에서의 연산 수행

- 이제 모델 정보를 읽고 빠른 연산을 위한 C++ 내 실행 등의 작업을 수행해야 한다. 

- Pybind 를 통한 연산의 C++ 구현 

    - 우선 연동부터 했음, c++ 에서 작성한 모듈 들을 Pybind 를 통해 pyd 파일로 변환, 해당 pyd 파일을 import 하여 작성 모듈을 사용한다잇~ 09/02

    - 이렇게 작성한 pyd 파일은 keras 의 backend 형태로 사용하자

# 3. BackEnd 연산 단위를 그럼 어디서부터

- 일단 가장 기본적인 행렬 덧셈, 행렬 곱 - 09/02

- 활성화 함수 (relu, sigmoid, tanh, leaky_relu, softmax)

- 일단 각 layer 의 종류별로 call 메서드에서 백엔드 메서드를 직접 호출하는 방식을 사용해 구현하자

- back propagate operations, 미분 계산 그래프 만들기

    - 모델 구성과 함께 만들어져야 하는데

    - 계산 그래프가 C++ 에 전달되고 어떻게 읽을지 고민 X 계산 그래프가 전달되는 형태가 아니고 계산이 수행되면서 계산 그래프가 완성되는 것, 그러면 실제 연산을 바로 layer, call 메서드에서 수행할 지...

        - 어휘 분석기 형식으로 계산 그래프를 만들어보자. - 트리 형태로 전달 X

        - 계산 그래프는 파이썬에서 만들어진 후 넘겨지도록 X

        - 먼저 딕셔너리 형태로 저장되어 있는 모델 정보가 C++ 에서 어떻게 읽어들이는지 확인해보자

        - 아니면 딕셔너리 정보에 따라 C++ 함수 호출 방식의 변경으로 진짜 C++ 에서는 연산만 수행하도록 하는건??

# 4. 모델 정보 저장 및 전달

- 단순 딕셔너리 형태로 layer 종류, 가중치, activaton function, input_shape 가 저장되어도 C++ 에서의 처리는 크게 상관없음

    - 가중치는 build 되는 레이어에 저장되는 것이 맞음, (이전 레이어의 output_shape 혹은 units 수 를 불러오면서 )

    - 연산은 layer 의 call 메서드에서 실행되어야 한다 여기서 backend 호출 

    - activation 이 그냥 python 연산으로 수행됨

    - 

- add 단계까지는 완료! - 09/03

- model.compile 부분 구현해보자, optimizer, loss, metrics 등등

    - optimizer 먼저 해보자잇 - 09/03

    -