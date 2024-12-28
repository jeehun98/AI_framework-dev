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

    - loss function 의 구현 - 09/03

    - metrics 까지 완료~09/03

        - 여러 종류의 compile parameter 를 앞으로 추가해야 한다.

- model.fit 을 구현해보자!!

    - x, y, epoch 세 개의 파라미터만 먼저 해보자

        - 가장 먼저 초기화된 가중치와 입력값의 연산, 역전파 없이

        - 데이터 입력을 numpy array 로 해야하는 문제가 발생 backend 부분의 수정 필요

        - backend C++ 파일 수정, flatten 및 dense 연산 수행 확인 - 09/03

            - Flatten 의 input_shape 수정 필요

        - activation 연산 확인

        - 데이터 하나씩 처리하는 것이 아닌,전체 데이터 연산이 가능하도록 변경 완료 - 09/04

- model.compile 의 loss, metrics 구현

    - loss (비용 함수) 에서 다차원 입력을 받을 수 있도록 구현 - 09/04

    - metrics 완료, 필요한거 더 추가하기 - 09/04

# 5. optimizer 구현 전, 역전파의 구현

- 드디어!! 계산 그래프  

    - layer 의 호출이 일어날 때마다 Node 객체가 생성된다. 

        - 노드의 생성은 C++ 백엔드 연산 내에서 구현하도록 해야겠다 - 09/05

        - 행렬 곱에서의 노드 생성

            - 각 원소별 곱셈을 수행하는 노드와 그 곱셈의 합을 수행하는 노드 간 부모 - 자식 연결

            - (m, n) (n, l) 의 행렬 곱 수행 시, m * l 개의 덧셈 노드와, 각 덧셈 노드는 n 개의 곱셈 노드를 포함하고 있다. - 09/05 완료

        - activation 연산 backend 내 구현 Node 연결

            - 활성화 함수 연산들의 구성 요소들을 노드 연산 상 구현을 해야하네

                - relu 의 경우, compare - select 노드, compare 는 gra 값 변화가 없고, select 는 output / input 의 gra 값을 가짐

                - sigmoid, softmax 까지 구현 완료~ - 09/09

        - 배치단위 입력에 대한 노드 출력이 1차원 리스트로 변환됨, 행렬 곱도 마찬가지, 각 행렬 요소들에 대한 노드 리스트로 n 개의 배치의 경우 엄청 긴 1차원 리스트가 출력되어 맞게 끊어주고 작업 필요

    - 각 layer 에서 생성된 노드 계산 그래프를 연결해야 한다. 

        - 행렬 연산, backend/operaters 출력 형태 통일하기

        - 기울기 계산 및 갱신을 하기 위해서는 모든 노드의 입, 출력값과 노드의 operation 별 정의된 gra 값이 필요

- 역전파 연산 구현하기

    - 각 layer 에서 생성된 노드들의 연결하기

        - hot issue 발생, flatten 층의 input_shape 가 없을 경우!! 비상 비상 

        - 일단 fit 연산의 metrics 까지 구현했음, 이제 node 를 연결하는 작업을 해야 함 - 09/09

        - dense 층에서 연산과 함께 받아오는 node_list 에서 연결하는 방법

            - 행렬 곱, 행렬 덧셈 이후 2차원 배열 형태로 변환은 하지 않음, 행렬 형태 변환에 필요한 정보들은 이미 다 있으므로 - 09/09

            - flatten 의 call 함수의 반환 값은 node_list 가 아닌 None 값으로 지정

            - 각기 다른 layer 의 node_list 를 연결하거나, 부모 자식 관계를 지정해줘야 한다

                - 새로운 node_list 가 입력될 때는 해당 값은 가장 상위 부모 노드에 해당, 하지만 이전 node_list 도 가장 상위 부모 노드만 있으므로 새로운 node_list 의 자식 노드까지 방문 후, 해당 노드와 이전 node_list 의 값과 부모 자식 관계를 연결해줘야 한다. 

                - 개별 레이어에서 수행해봐야겠다.

                - trainable 레이어를 지정해봐야지

    - node 의 ops 별 gra 값을 지정해주자 

        - 사칙, exp 구현 완료 - 09/10

    - loss 의 gra 값을 지정하기 위해 이것도 계산 그래프를 형성해야겠다. 지금까진 Dense 만 구성했음

        - mse 의 노드 그래프 형성 - 09/10

        - Cross-Entropy Loss 노드 그래프 형성 완료 - 09/10

        - node_list 를 전달하는 방법으로 연결하는 것도 좋아보이네

    - 이제 구성된 계산 그래프의 최상위 루트 노드인 loss_node_list 에서 시작해보자잇 - 09/10

        - node 클래스랑, activation 클래스랑 따로 관리하자...

        - loss_node_list 를 분리해주자.. 개별 데이터에 대한 가중치 갱신 값을 먼저 구하기 위해!!

            - 2차원 numpy array 로 변환했어 일단~ - 09/10

            - 한 배치의 학습 -> 한 데이터의 학습 순서대로 해야해~ 

            - 역전파 과정에서 layer 를 거꾸로 읽으면서 해당 최상위 부모 노드와 loss 에서 계산한, 각 노드별 변화량에 대한 비용 함수의 변화량 연결해보자 처음 예시에선 출력 노드가 2개임

            - 배치를 전부 처리할 지, 데이터 하나를 먼저 처리할 지 고민

            - 아니 근데 그냥 노드에 저장되어 있는데?

            - 모든 리프 노드를 얻도록 코드 수정 - 09/10

            - 배치 단위 데이터들을 그냥 쭉 일차원으로 늘려서 붙이자, 차이는 없어

            - grad_a, grad_b 의 grad_input, grad_weight 의 명칭 변환

            - 조건문을 통해 특정 노드에 grad_input, grad_weight 중 어느 값을 전달해야 할 지 정한다.

            - activation 의 연산의 역전파 시 grad_input 값만 갱신됨

    - node 클래스를 정의하자 - 09/11

        - bias 덧셈 시 차원에 맞게 늘리지 말까...

        - 필요 기능 완료 - 09/11

        - backpropagate 메서드에서 grad_input, grad_weight 적절히 저장

            - 연산 별로 나눠서 구현해야함

            - multiply 의 경우 input_a, input_b 의 곱셈, 해당 값들이 하위 노드의 출력값인 경우가 있는지 생각해보자.

            - 행렬 곱에서의 생각을 해보면, input_a 는 은닉층 유닛의 입력값, input_b 는 해당 입력에 대응되는 가증치 값

            - 대응되는 가중치가 없는, grad_weight = 0 의 조건별 연산을 수행할 필요 없이 해당 노드에 필요한 정보가 저장되어 있다.

        - grad_input, grad_weight 를 계속 누적해야 하는 이유가 있을지

            - 계산 그래프는 모델 당 하나로 구성됨, 여기에 들어가는 데이터만 달라지는 것
            
            - 때문에 하나의 계산 그래프에서 grad_weight 을 누적하고, 배치 크기로 나눠 각 가중치 변화량에 대한 비용 함수의 변화량의 평균으로 갱신

            - model.compile 부분에서 계산 그래프를 초기화 해야함...

            - 계산 그래프 초기화 없이, layer 별 root 노드 전달을 통해 진행할거임

                - operations_matrix 완료 - 09/12

                - 

            - Dense 간 연결과 loss_function 연결을 따로 구현했음, Dense 층의 경우 행렬 원소에 대해 연결되기 때문

            - grad_input 의 경우 값의 누적이 필요하지 않고, 단순히 계산한 결과를 하위 자식 노드로 전달하기만 해도 됨, grad_weight 는 가중치 변화량에 대한 비용함수의 변화량으로 그 값을 누적한 후 배치 사이즈로 나눠 가중치 변화량의 계산 - 09/19
        
        - weight_update 구현

            - 각 layer 에 저장된 가중치 값을 어떻게 호출할까...

            - node 값에 weight 를 연산과 함께 저장을 해주자

            - 가중치가 내가 생각하던 모양대로 저장되어 있진 않네

            - 일단 multiply 연산의 가중치 갱신부터 해보자... - 09/19 완료

        - node.h 내에서 역전파 연산을 수행하도록 이전에는 어떻게 했지?

    - 전체 데이터를 배치 단위로 사용한다고 가정해보자

        - 노드 리스트가 업데이트 되는 것을 확인할 수 있다. - 09/19

        - layer 에 행렬 형태로 저장되어 있는 가중치, 만약 노드에 가중치 값이 있다면 해당 값과의 연산을 수행하도록 바꾸자
        
        - 각 epoch 가 끝난 후 가중치 업데이트

        - 생성된 계산 그래프에서 연산 시켜보자

    - node.h 에서의 역전파와 calculate_gradient 수행

        - node.py 에서 backpropagate 함수 내에서 자식 노드탐색 방문을 수행하는데 이를 node.h 에서 하는게 더 나으려냐

    - 순전파, 추론 단계에서는 계산 그래프를 사용하지 말자, 단순히 값을 넣기만 하고 - 09/20 

    - sequential 에서 가중치 갱신을 수행하게 되는데 node.h 에서 이를 수행하도록 하는 방법이 있을까?? - 09/24

    - weight_update 에서 무한 루프 시행됨 - 09/24

        - weight_update 의 경우 배치 데이터의 예측 및, 역전파가 끝나고 수행되어야 함,

        - 계산 그래프에 접근하는 방법이, sequential 모델의 self.node_list 값을 통해 root_node 에 접근한다. 

        - node.h 파일의 backpropagate 수정하기 상위 부모 노드에서 아래로 내려가기 완료 - 09/25

        - weight_update 전후의 노드 계산 그래프 출력해서 잘 변했는지 확인해보자

        - weight_value 를 출력하는데 바뀌질 않는 문제 - 09/25

        - weight_update 이후  weight_value 값이 진짜 엄청 커지네 

        - weight_update 과정을 출력하여 확인할까...
        
            - grad_weight_total 의 값에 의해 weight_update 가 시행되는데, 배치 사이즈크기로도 나눠줘야 한다. 

            - 첫 추론에서 임의의 가중치를 입력받아 연산을 수행 

            - 이후 노드 값에 가중치가 저장되어 있는 경우 저장되어 있는 가중치를 통한 연산을 수행해야 한다. 

            - dense.py 와 operations_matrix.cpp 의 데이터 전달 확인, 노드 리스트가 전달될 경우...

                - operations_matrix 도 수정해야 한다. 파이썬으로 부터 입력값과 가중치를 전달받아 연산을 수행후 노드 그래프에 저장하는 방식임 지금

                - 수정할 방향으로 이미 node_list 가 존재할 경우 해당 노드 list 로 부터 가중치 데이터를 전달받아야 한다. - 09/25 완료

        - epoch 가 끝난 후 비용 함수값 확인 - 유의미한 값은 아니네...

            - learning_rate 를 더 낮췄을 때, 그 값의 변화가 적어져야 함 - 이건 맞음 09/25

            - 배치 데이터들의 가중치 변화량 합의 평균을 사용해서 그렇군...

    - 실제 데이터 셋, 당뇨병 데이터 ( 10 개의 실수 특성, 연속형 타겟 변수 ) 사용

        - epochs 를 적용시켜보자!         
        
        - epoch 가 끝난 후 마지막으로 바뀐 가중치에 대해 연산을 수행, 이는 predict 와 동일한 기능 (비용 함수 값이 추가...)

            - 두 가지 함수를 추가해보자. 

            - predict, cal_loss 완료 - 09/26

        - 가중치가 변하지 않는 문제가 있었음, loss_list 에 대한 접근 설정을 하지 않았구나. 기존 계산 그래프가 있으면, 데이터만 수정하는 거...

            - loss 내의 노드 리스트 입력 시 구별 기능 구현, - 09/26

            - loss 에 예측 결과, 타겟 리스트의 전달, 완료 - 09/26

# 6. optimizer 구현하기

    - optimizer 에서 weight_update 수행하기, 지금은 weight_update 의 SGD 방법을 사용한 가중치 갱신을 사용하고 있음 node.h 내 구현

    - optimizer 코드 내에서 루트 노드에서 시작하여 node 의 가중치, 업데이트, grad_weight_total 정보의 사용, 

    - grad_weight_total / (learning_rate * batch_size) 의 값이 가중치 갱신량으로 사용, SGD 에선 단순히 그 값을 빼서 weight_update 수행

    - optimizer 객체를 무조건 생성해서 사용해야 하네...

        - 인스턴스의 생성 확인 - 09/26

        - 인스턴스의 attribute 호출 오류 문제 확인, 바인딩 및 cpp 코드 상 문제가 있을 거라고 생각된다. 

        - 문제 해결, learning_rate 도 새로 입력받아서 사용하는 방법 적용 완료 - 09/27

# 7. Dense 과정 다 완료

    - Sequential 모델 선언, layer 추가, compile 을 통한 세팅, fit 으로 학습

# 8. CNN 구현하기

    - Conv2D 의 구현

        - 가중치 생성 build(), 합성곱 연산 수행 call() 

        - Conv2D.cpp 파일 작성, 패딩도 여기 안에서 수행하도록

        - 계산 그래프가 어떻게 생성되는지 생각해보자. 

        - call 연산에서 전달되는 형태의 문제 발생

        - 가중치 생성 시 입력 데이터의 차원 수가 필요하다

        - 차원 수 맞추기 완료

        - pooling_layer 구현하기 10/04 시작

            - max 와 mean pooling 구현

            - 연산의 과정
            
                - pool_size 를 돌아다니면서 값의 갱신 - 10/04 완료
            
            - pooling 이후 계산 그래프 연결 과정 추가를 생각해보자

                - pooling node 연결 로직을 추가해보자.

                - link_pool_node 구현

                - current, previous_layer 의 변경 구현 완료 - 10/05

        - conv 의 input_shape 에 대해 생각... 사용자가 직접 입력하긴 어려움, 모델 내에서 계산해 추가하는 로직을 생각해야 하나.

            - 각 layer 의 input, output_shape 의 지정, 가중치가 없을 경우 이전 레이어의 값을 호출하는 로직의 추가

            - layer 클래스의 input_shape, output_shape 완성시키기

                - model.add 과정에서 추가하는 로직 구현 sequential 수정하기

                - sequential.add 과정 수정중

                - activation 연산 결과가 list 인 것을 수정하기, numpy_array 변환 완료 - 10/07

                - conv2D 에서 출력 차원의 크기를 미리 계산하는 call_output_shape 를 구현하기

                - pooling 의 strides 값 형식 변경, layer 종류별 output_shape 의 구현 완료 - 10/07 

        - conv2D layer 에 Flatten, Dense 층의 추가 형태 구현 확인

            - 가중치 생성 문제 발생, 해결 완료 각 layer 의 input, output_shape 의 재지정 완료 - 10/07

        - node 의 연결 문제, seqential 코드에서 link_node 가 잘 작동하는지, self.node_list 를 업데이트할 지에 대한 고민

            - node_list 연결 중, 두 번째 연결부터 이상한데 이걸 해결해보자

                - root_node, leaf_node 간 연결 문제일 수도 있음

        - loss 출력 맞추기 - 이걸 하기 위해서는 비용 함수를 추가해야해 이전까지는 binary cross entropy 만 구현했어

            - (1,5) 형태의 예측과 (1,1) 의 타겟값, 각 타겟에 대한 확률값을 출력한 것

            - 추후에 구현 같이해야해

            - 까먹지 말기 - 10/07 ~~~ 킵고잉

    - backpropagate 도 구현해야해

        - node_list 연결 문제 발생!! 어디서 문제인건지\

        - 필터 가중치, (3,3,14,28) 에서 각 필터가 (3,3,14) 크기의 가중치, 28개가 존재

        - link_node 의 수정 필요

            - 지금까지 세 가지 조건문, layer 종류별 link_node 수행,  dense, activation, pooling 층만 구현

            - link_conv2d_node 를 구현하자

                - pooling 내애서 연결을 구현할 수 있을 듯
        
                - pooling.cpp 코드 상의 문제 먼저 해결해보자. max 가 제대로 작동하지 않음

                - layer_node_list 를 업데이트 해야 할 지

                - pooling.cpp 내의 반복문 순서의 차이로 인한 오류였음, 수정 완료 - 10/10

                - 연산을 잘 수행해보자.

                - link_conv2d_node 제대로 구현된 것을 확인

                - cpp 파일 이상이 있는지 확인해보자

                    - 특히 pooling, relation_node 가 이상하게 보임

            - flatten 층 연결하기

                - flatten 의 node_list 를 구현할까

                - node_list 구현 완료, trainable 의 제거 - 10/10

        - multiply 가 증가하는 것은 첫 번째 레이어의 경우 필터 크기 * 입력 차원의 수, 그 다음 layer 에 대해선 증가된 차원에 대한 연산이 추가, 각 필터는 입력 차원 * 필터 크기에 대해 연산을 수행한다. 

# epoch 별 반복 변환

    - 각 epoch 시 사용되는 배치 데이터 셋이 달라져야 함 - 09/30 구현해야 해 

# pytorch 와 비교해서...

    - pytorch 에서 모델의 출력인 output 에 계산 그래프가 포함되어 있음, 해당 값을 사용한 loss 연산을 통해 계산 그래프가 연결

    - 이전에 optimizer 에서 사용된 model.paramters 들로 어떤 값들이 해당 가중치 갱신 방법들을 사용할 것인지에 대한 선언이 수행된다.

    
# activation_layer 의 추가

    - Dense layer 에선 activation 오브젝트를 지정받아옴.

    - model.add(Activation('softmax)) 와 같이 activation layer 를 추가함

    - __init__, call 구현 완료

    - Sequential 에서 add 부분의 로직을 추가해야해, input_shape 를 추가해줬음 - 완료 10/01

    - output 을 리스트로 감싸주면서, Dense 층과의 연결은 되었지만, loss 와의 연결 및 연산의 문제 발생

        - numpy_array 로 변환 과정 추가를 통한 문제 해결 - 10/04

# 후진 모드 자동 미분을 위한 계산 그래프 연결

    - activation_layer 가 추가되면서 여기에서도 계산 그래프 연결 로직이 필요해졌음.

    - 이전에는 이러한 로직이 loss_function 과 계산 그래프를 연결하는 과정을 따로 추가했지만, layer 를 따라 model.fit 연산을 수행하는 과정에서 layer 의 종류와는 상관없이 계산 그래프를 연결하는 로직이 추가되어야 함

    - 각 layer 에 self.node_list, trainable 속성의 추가와 초기 계산 그래프 생성시 해당 레이어들을 연결해주는 로직을 추가, 완료 - 10/01

    - activation_layer 연산에서 노드 리스트 문제

    - layer 의 종류별 link_node 를 다르게 구현해야 하네

        - 먼저 activation_layer 인 경우 일대일 노드 리스트 연결 수행

        - activation function 의 call 연산의 결과를 감싸줘서 해결 완료 - 10/02


# sequential Class

    - 배치 데이터 나누는 로직을 추가하자

    - pytorch 에서는 DataLoader 를 통해 구현했음, 

    - 구현 완료 - 10/02

# activations, losses, optimizers 추가하기 - 10/02

    - 먼저 activation 부터 하자..

# RNN 구현하기 - 10/10

    - 가중치의 종류와 shape 에 대한 이해, 동일한 하나의 가중치가 사용된다는 걸 이해했음

    - recurrent.cpp 작성중 

        - activation 의 node 도 작성해야함

        - rnn 연산 과정 확인하기, recurrent 의 과정이 어떻게 수행되는지 보자 보자

        - 맨 처음 add node 가 생성되는 이유, 이전 은닉층의 출력이므로 h_0 임

        - 트리 구조가 어떻게 구성될 지 생각,    

            - 내가 오해하고 있었음, n 차원 데이터로 벡터화된 각 토큰들, 각 차원별 다른 가중치들이 사용된다.

            - 유닛의 개수는 각 타임 스텝에서의 은닉 상태

            - 가중치 생성 부분을 다시 확인해보자. - 10/14 입력 벡터 차원과 은닉 차원에 대해 정확히 이해했음, 수고링

            - RNN 의 연산 구조를 따라 트리를 구성할 경우 순환 이 적용됨, 각 타임 스텝의 은닉 상태를 복사하면 단방향 트리 모델을 만들 수 있음

            - node_list 가 있을 경우의 조건문 형식 올바르게 수정

            - 각 루트 노드에 접근하기

                - 다음 은닉층의 입력에 활성화 함수 적용값 다음 은닉층의 입력으로 사용,

                - activation_node 가 루트노드가 되는게 맞겠다.

                - 활성화 함수 연산, 노드 연산 가져오기

                - 컴파일 명령어에 activation.cpp 파일의 추가, g++ -shared -m64 -I"C:\Users\owner\AppData\Local\Programs\Python\Python312\include" -I"C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include" -L"C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib" binding_recurrent.cpp ../activations/activations.cpp -o recurrent.pyd "C:\Users\owner\AppData\Local\Programs\Python\Python312\libs\python312.lib"   

                - activation_node, bias_node 연결 수정

                    - node.h 에 필요한 함수를 구현하자

                - recurrent.cpp, node 연결 수정하기 - 완료 10/14

            - hidden_unit 의 개수가 4 개 이상일 때 생기는 오류...

            - 입력 데이터의 차원과 길이에 대한 차이를 이해하자

        - 아 이해가 안가네 왜 unit 의 개수가 특정 개수 이상일 때 오류가 나는 이유가 뭘까...

        - operations_matrix 를 통한 RNN 연산 수행

            - 계산 그래프가 이미 만들어져있는 경우 node_list 를 어떻게 전달할 것인가

            - 전달되는 node_list 는 히든 유닛의 출력, 그 길이는 유닛의 개수와 동일

            - operations_matrix 에서 받는 데이터의 크기가 바뀌지 않았음 전처리 시행하자

            - 잘못된 node_list 의 전달 문제

            - recurrent.cpp 내에서 node_list 연결을 수행해야 할 지에 대한 고민

            - 각 타임 스텝에 대한 반복, 동일한 계산 그래프를 사용하는 것이 아님, 계산 그래프가 계속 이어져서 생성되어야 함 - 구현 완료 10/16

                - 각 node_list 를 연결해줘야하기 때문에 각 상태 이후 node_list를 다른 이름으로 저장하고 있음

                - is_new_graph 조건문을 없애는 것부터 시작, 구현 완료 후 기존 모델들의 수정을 시행할거야

                - input_multiply_node_list, recurrent_multiply_node_list 두 노드 리스트의 길이는 동일

                - bias 의 값은 input_multiply, recurrent_multiply 의 합에 추가로 더해짐

                    - operations_matrix 의 add 부분 수정하기

                        - 여길 수정하진 않고 recurrent.cpp 부분에서 노드 연결 부분 구현했음 - 10/16

            - 벡터 투 시퀀스 개념으로 기본 모델 구현, 반한되는 node_list 는 마지막 타임 스텝에 대한 연산 결과로 활성화 함수의 연산 결과, activation_node_list 임 

                - 마지막 타임 스텝만 출력 완료 - 10/17

    - RNN layer 에 대해서 출력 shape 를 계산, build 단계에서 output_shape 를 지정해줬음

    - return_sequences 구현하기

        - 각 타임 스텝의 출력을 사용해야 함 

        - recurrent.cpp 내에서 조건문의 추가, 

    - rnn 구현 내용 오류 발견

        - 계산 그래프 출력해서 보는데 recurrent 가 제대로 되어있지 않음, 일렬로 쭉 나와야해

        - 순환 가중치 연산의 입력값을 이루는 노드들에 자식 노드가 포함되어 있어야 함

    - result 의 값과 root_node 의 output 이 다른 문제 - 10/21

        - 해결 완료 - 10/21

    - rnn 역전파 과정과 

# 배치 학습 내용 수정

    - stochastic 의 경우 오류 발생 - 10/17

        - 계산 그래프의 입력이 더 이상 필요하지 않음 지금,

        - node_list 의 조건문을 없애자, self.node_list 를 넘겨주지 않을거임

        - 이에 따라 바뀌는 내용이 추가로 뭐가 있을지 생각

        - 따로 뭐 있진 않네 해결  - 10/17

# 계산 그래프에 대해 상기

    - 계산 그래프를 구현하고, 계산 그래프의 입력 유무에 따른 차이를 두는 이유는, 가중치 갱신에 있어 stochastic 이 아닌 batch 데이터를 사용하는 경우에 있어 기존의 계산 그래프를 재사용하는 것

        - 이 경우 사용되는 것이 배치 데이터의 각 loss 값의 평균을 통한 것일텐데

        - 가중치 갱신량의 평균이 아님...

        - 계산 그래프를 매번 다시 생성하는 것이 나을 수 있겠다... 기존 dense 의 구현 내용을 확인해봐도...

    - sequential.py 에서 오류 발견, self.node_list 의 갱신이 없었음... - 10/16

        - 1 epoch, 1 data_set 의 조건문도 없애자...

        - node_list 도 입력받을 필요 없음

    - 로직을 다 바꾸자 is_new_graph 의 조건문이 필요없어졌어.

        - 가중치 갱신 이후 누적되어야 하는 값이 없어

    - 가중치 갱신 연산 이후 layer 의 weight 값의 update

    - 각 부분을 더 작게 나눠서 구현할지

        - dense 는 bias, activation 을 밖에서 불러와서 연결 연산 수행

        - rnn 는 안에서 다 했음

    - backpropagate 에서 visited 부분의 삭제 - 10/21

    - 배치 데이터의 경우 각 데이터 별 loss 값에 대한 평균을 통해 가중치 갱신을 수행하므로 하나의 계산 그래프만 있어도 됌 - 10/21

        - root_node 의 개수는 binary, categorical, 종류에 따라 그 개수가 달라짐 다중 분류의 경우 가중치 갱신량은 각 category 별로 누적되어 연산됨

        - 

# layer 쌓기에 더해서 연산을 쌓아보자

    - operation 을 쌓기 위한 새로운 model, operaion_combination_layer, OCL 을 구현

    - keras 구현 내용
    
        - x = layers.Dense(64, activation='relu')(inputs) 

        - 사전 정의된 layer 를 사용

        - 동일하게 사전 정의된 연산자들이 있어야 함

        - Node 단위 구현을 해야 하는데, 

        - activation function 을 적용하는 데 있어 이는 layer 를 쌓는 것과 동일,

            - 때문에 먼저 layer 단위 함수형 API 를 구현해야 하네

    - x 는 단순한 값이 아닌 객체임... 클래스를 정의해야 함

        - 심볼릭 텐서를 만들어야 해

            - 심볼릭 클래스 정의를 한다. 

    - InputLayer, 또는 input 의 사용!! 두 가지의 차이도 인지

        - output 속성이 명시적으로 정의되지 않음

    - call 연산을 수정해야 해 여기서 난 그냥 연산이 이뤄지는 줄 알았는데, 그게 아니라 입력 데이터에 따라 계산 그래프 구성 혹은 연산이 이뤄지는 것

    - 가장 먼저 input_layer __call__ 연산의 반환 객체 정의 - 11/06

        - node 객체를 반환할 것, c++, python 두 환경에서 모두 구현할 계획

        - c++ 상에서 구현한 node 클래스를 python 에선 nodeWrapper로,

            - 

        

# CUDA 구현

    - CUDA 설치 및 실행 완료, GPU 정보 확인 - 10/22

    - json 파일 상의 문제인지, 컴파일러 정보를 못 가져오는 문제 발생

        - Native tools 를 사용하면 문제가 없음, 컴파일 file_path 나 컴파일러 자체의 문제일 것 (cl.exe 를 왜왜왜왜오 못 불러오니...)

        - 일단 외부 도구를 사용해서 빌드 및 실행을 확인하자

    - pybind, cuda 코드 연동 확인 - 12/19

    - python 과 cuda 의 역할 분리 수행 시작 - 12/23

        - Node Class 의 분리 수행


# notion 작성중

    - cal_dense.py 확인 중, build 부분의 input_shape 수정 필요 - 11/08 - 완료 build 시 self.input_shape 지정 문제

        - 노드 리스트 연결 문제 - 11/08

        - dense.py 에서 call 연산에서 자식, 부모 연결이 잘못되었음, 연결없이 노드 리스트 업데이트 수행했네

        - Dense Layer 의 bias, act 연산 부분 수행 완료

        - layer 입력 데이터 형태 문제, dense call 출력형태 바꿔줬음 - 11/11

    - 각 layer 클래스와 함께 backend 쪽 구현, 연결 부분도 같이 표시해줄지에 대한 고민,

        - Dense 와 RNN 의 연결 방식 통일 방법 고민

    - cnn 연결문제 - 11/20

# fortpolio 작성

    - 마인드 맵을 구성해야해

        - 처음 예시로 시작하여 어떻게 진행되고 처리되는지, 이후 추가된 로직같은

    - Pybinding11 문서 작성, 그래프 추가 필요 - 12/17

    - 사전 정의된 연산들의 조합을 사용하여 빌트인 레이어의 생성

    - 사전 정의된 연산이란 연산의 미분값이 포함되어 있는,  