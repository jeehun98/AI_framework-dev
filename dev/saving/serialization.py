import json

# 직렬화 함수 구현
def serialize(activation):
    if isinstance(activation, CustomActivation):
        return {
            "class_name": activation.__class__.__name__,
            "config": {
                "name": activation.name,
                "param": activation.param
            }
        }
    elif callable(activation):
        # 함수나 람다일 경우에는 이름을 반환 (함수 이름으로 직렬화)
        return activation.__name__
    else:
        raise ValueError("Unknown activation type. Cannot serialize.")

# 역직렬화 함수 구현
def deserialize(config, custom_objects=None):
    if isinstance(config, str):
        # 함수 이름으로 제공되었다면, 해당 함수 찾기
        if config in globals():
            return globals()[config]
        elif custom_objects and config in custom_objects:
            return custom_objects[config]
        else:
            raise ValueError(f"Unknown activation function: {config}")
    
    elif isinstance(config, dict):
        # 클래스 이름과 설정으로 제공된 경우, 해당 클래스 복원
        class_name = config.get("class_name")
        if class_name == "CustomActivation":
            return CustomActivation(**config["config"])
        else:
            raise ValueError(f"Unknown class name: {class_name}")
    else:
        raise ValueError("Unknown config format for deserialization.")

