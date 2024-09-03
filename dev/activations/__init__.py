from dev.activations.activations import sigmoid
from dev.activations.activations import relu
from dev.activations.activations import leaky_relu
from dev.activations.activations import softmax
from dev.activations.activations import tanh

# 활성화 함수 객체와 딕셔너리
ALL_ACTIVATIONTS = {
    sigmoid,
    relu,
    leaky_relu,
    softmax,
    tanh,
}

ALL_ACTIVATIONS_DICT = {fn.__name__: fn for fn in ALL_ACTIVATIONTS}

# layer 에서 activation 이 설정되고 사용될 때, 해당 get 메서드를 통해 
# parameter 에 해당되는 activation 의 각 메서드가 저장됨
"""

# 'relu'라는 이름으로 활성화 함수 가져오기
activation_fn = get("relu") # 실제론 문자열 자체가 아닌, 파라미터 값으로 호출하게 될 것

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = activation_fn(x)

"""
def get(identifier):

    # 검증하는 역할, 사용자가 파라미터로 정의되지 않은 활성화 함수의 입력을 했을 경우가 존재
    if isinstance(identifier, str):
        obj = ALL_ACTIVATIONS_DICT.get(identifier, None)

    if callable(obj):
        return obj
    
    raise ValueError(
        f"Could not interpret activation function identifier: {identifier}"
    )