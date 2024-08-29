class SomeClass:
    def __new__(cls, *args, **kwargs):
        print("Creating instance")
        obj = super().__new__(cls, *args, **kwargs)
        return obj

    def __init__(self, *args, **kwargs):
        print("Initializing instance")

# 객체 생성
instance = SomeClass()

print(instance)