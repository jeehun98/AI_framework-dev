from dev.activations.activations import sigmoid, relu, leaky_relu, softmax, tanh

# 활성화 함수 객체 집합
ALL_ACTIVATIONS = {
    sigmoid,
    relu,
    leaky_relu,
    softmax,
    tanh,
}

# 활성화 함수 이름과 객체를 매핑한 딕셔너리
ALL_ACTIVATIONS_DICT = {activation.__name__: activation for activation in ALL_ACTIVATIONS}

def get(identifier):
    """
    활성화 함수의 이름(identifier)을 문자열로 입력받아, 
    해당 이름에 맞는 활성화 함수 객체를 반환합니다.
    
    Parameters:
        identifier (str or callable): 활성화 함수의 이름(str) 또는 함수 객체.
    
    Returns:
        function: 호출 가능한 활성화 함수 객체.
    
    Raises:
        ValueError: 유효하지 않은 이름이 입력된 경우 예외를 발생시킵니다.
    """

    # 입력이 문자열인 경우, 딕셔너리에서 활성화 함수 가져오기
    if isinstance(identifier, str):
        activation_fn = ALL_ACTIVATIONS_DICT.get(identifier)

        # 함수 객체가 유효한 경우 반환
        if callable(activation_fn):
            return activation_fn

    # 입력이 이미 함수 객체인 경우, 그대로 반환
    if callable(identifier):
        return identifier

    # 유효하지 않은 입력일 경우 예외 발생
    available_activations = ", ".join(ALL_ACTIVATIONS_DICT.keys())
    raise ValueError(
        f"Invalid activation function identifier: '{identifier}'. "
        f"Available options are: {available_activations}."
    )
