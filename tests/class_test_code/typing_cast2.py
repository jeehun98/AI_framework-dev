class Parent:
    def __new__(cls, *args, **kwargs):
        print("Parent __new__ called")
        instance = super().__new__(cls)
        return instance

class Child(Parent):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        print("Child __new__ called")
        return instance

child = Child()

"""
Parent __new__ called
Child __new__ called
"""
print(type(child))

print(isinstance(child, Child))  # 출력: True
print(isinstance(child, Parent)) # 출력: True
print(isinstance(child, object)) # 출력: True
